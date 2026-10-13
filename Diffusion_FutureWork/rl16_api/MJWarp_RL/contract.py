"""Shared, hardware-independent contract for independent hard-tip control."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Iterable
import zlib


CONTRACT_SCHEMA = "mjwarp-g6-independent-hard-tip-contract-v22"

OBJECTS = (
    "orange",
    "banana",
    "pear",
    "strawberry",
)
INFEASIBLE_OBJECTS = frozenset(("cylinder", "cone"))
PLATE_OBJECT = "plate"
RIGID_OBJECTS = frozenset()
FRUIT_OBJECTS = frozenset(("orange", "banana", "pear", "strawberry"))

TASK_TRANSLATE = "translate"
TASK_ROTATE = "rotate"
TASKS = (TASK_TRANSLATE, TASK_ROTATE)

TRANSLATION_FRAME = "post_lift_action_reference_world_xy"
TRANSLATION_DZ_M = 0.0
ACTION_REFERENCE_CAPTURE = "post_lift_measured_com_and_commanded_cage"
# Translation samples exactly one of five canonical paths.  Their geometry is
# centered on the 100 mm activity disk and is invariant across curriculum
# phase, seed, object, and platform.  Twenty equal-arc-length intervals retain
# the existing compact 21-anchor observation only.  Runtime targets follow a
# separately certified, acceleration-bounded arc-length schedule and never
# restart minimum-jerk motion at those observation anchors.
TRANSLATION_PATH_FAMILY = "centered_fixed_five_minimum_jerk_v3"
TRANSLATION_PATH_REGION_RADIUS_M = 0.100
TRANSLATION_PATH_SEGMENTS = 20
TRANSLATION_PATH_WAYPOINTS = TRANSLATION_PATH_SEGMENTS + 1
TRANSLATION_PATH_SHAPES = (
    "square_100mm",
    "circle_r50mm",
    "letter_s_200mm",
    "letter_u_200mm",
    "letter_n_200mm",
)
# The retained seed field is a fixed provenance sentinel.  It no longer
# rotates, mirrors, translates, scales, or otherwise mutates path geometry.
TRANSLATION_PATH_SEED_MIN = 1
TRANSLATION_PATH_SEED_MAX = 1
TRANSLATION_SQUARE_SIDE_M = 0.100
TRANSLATION_CIRCLE_RADIUS_M = 0.050
TRANSLATION_LETTER_PATH_LENGTH_M = 0.200
TRANSLATION_U_RADIUS_M = 0.030
TRANSLATION_PATH_LENGTH_M_BY_SHAPE = (
    0.400,
    0.314159265359,
    0.200,
    0.200,
    0.200,
)
TRANSLATION_PATH_STEP_M_BY_SHAPE = tuple(
    length / TRANSLATION_PATH_SEGMENTS
    for length in TRANSLATION_PATH_LENGTH_M_BY_SHAPE
)
TRANSLATION_PATH_MAX_LENGTH_M = max(TRANSLATION_PATH_LENGTH_M_BY_SHAPE)
TRANSLATION_PATH_MAX_STEP_M = max(TRANSLATION_PATH_STEP_M_BY_SHAPE)


def translation_path_point_xy_m(
    path_shape_index: int,
    progress: float,
) -> tuple[float, float]:
    """Evaluate one canonical path at normalized arc-length progress."""

    if isinstance(path_shape_index, bool) or not isinstance(
        path_shape_index, int
    ):
        raise ValueError("path_shape_index must be an integer")
    if not 0 <= path_shape_index < len(TRANSLATION_PATH_SHAPES):
        raise ValueError("path_shape_index is outside the fixed path table")
    value = float(progress)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("path progress must be finite and inside [0, 1]")

    if path_shape_index == 0:
        side = TRANSLATION_SQUARE_SIDE_M
        half = 0.5 * side
        distance = 4.0 * side * value
        if distance <= side:
            return (half - distance, half)
        if distance <= 2.0 * side:
            return (-half, half - (distance - side))
        if distance <= 3.0 * side:
            return (-half + (distance - 2.0 * side), -half)
        return (half, -half + (distance - 3.0 * side))

    if path_shape_index == 1:
        radius = TRANSLATION_CIRCLE_RADIUS_M
        angle = 2.0 * math.pi * value
        if value == 1.0:
            return (radius, 0.0)
        return (radius * math.cos(angle), radius * math.sin(angle))

    distance = TRANSLATION_LETTER_PATH_LENGTH_M * value
    if path_shape_index == 2:
        radius = 0.100 / math.pi
        if distance <= 0.100:
            angle = math.pi / 2.0 + distance / radius
            point = (
                radius * math.cos(angle),
                radius + radius * math.sin(angle),
            )
        else:
            angle = math.pi / 2.0 - (distance - 0.100) / radius
            point = (
                radius * math.cos(angle),
                -radius + radius * math.sin(angle),
            )
        if value == 0.0:
            return (0.0, 2.0 * radius)
        if value == 1.0:
            return (0.0, -2.0 * radius)
        return point

    if path_shape_index == 3:
        radius = TRANSLATION_U_RADIUS_M
        vertical = (
            TRANSLATION_LETTER_PATH_LENGTH_M - math.pi * radius
        ) / 2.0
        center_y = (radius - vertical) / 2.0
        if distance <= vertical:
            return (-radius, center_y + vertical - distance)
        if distance <= vertical + math.pi * radius:
            angle = math.pi + (distance - vertical) / radius
            return (
                radius * math.cos(angle),
                center_y + radius * math.sin(angle),
            )
        return (
            radius,
            center_y + (distance - vertical - math.pi * radius),
        )

    scale = TRANSLATION_LETTER_PATH_LENGTH_M / 13.0
    width = 3.0 * scale
    height = 4.0 * scale
    diagonal = 5.0 * scale
    if distance <= height:
        return (-width / 2.0, -height / 2.0 + distance)
    if distance <= height + diagonal:
        along = distance - height
        return (
            -width / 2.0 + width * along / diagonal,
            height / 2.0 - height * along / diagonal,
        )
    return (
        width / 2.0,
        -height / 2.0 + (distance - height - diagonal),
    )


# These extrema are certified on the same 1e-12 canonical grid as the fixed
# tables decoded below.  Keeping them literal prevents platform libm drift
# from entering the algorithm payload during module import.
TRANSLATION_PATH_START_RADIUS_MAX_M = 0.070710678119
TRANSLATION_PATH_RELATIVE_EXTENT_MAX_M = 0.141421356237
TRANSLATION_PATH_GENERATION_RADIUS_M = TRANSLATION_PATH_REGION_RADIUS_M
TRANSLATION_PATH_LEAD_IN_TICKS = 300
TRANSLATION_PATH_SEGMENT_TICKS = 90
# Read-only compatibility aliases for pre-v19 artifact readers.  New runtime
# code and all v19 manifests expose one phase only.
PHASE1_TRANSLATION_PATH_SEGMENT_TICKS = TRANSLATION_PATH_SEGMENT_TICKS
PHASE2_TRANSLATION_PATH_SEGMENT_TICKS = TRANSLATION_PATH_SEGMENT_TICKS
TRANSLATION_PATH_START_SETTLE_TICKS = 300
TRANSLATION_PATH_TERMINAL_HOLD_TICKS = 1_802
FORMAL_ACTION_TICKS = 4_202
TRANSLATION_PATH_ROUTE_SAMPLES = (
    TRANSLATION_PATH_SEGMENTS * TRANSLATION_PATH_SEGMENT_TICKS
)
TRANSLATION_CONTROL_DT_S = 0.020
TRANSLATION_SCHEDULE_QUANTIZATION_DECIMALS = 12
TRANSLATION_SCHEDULE_QUANTIZATION_SCALE = (
    10**TRANSLATION_SCHEDULE_QUANTIZATION_DECIMALS
)
TRANSLATION_SCHEDULE_TABLE_HASH_ENCODING = (
    "sha256_big_endian_signed_int64_grid_1e-12"
)

# Match the proven FLUID motion envelope.  Both the centre-to-path lead and
# strict route are fixed-grid minimum-jerk schedules, so acceleration and
# braking approach zero continuously at every endpoint and sharp corner.
TRANSLATION_LEAD_HARD_MAX_SPEED_M_S = 0.040
TRANSLATION_LEAD_HARD_MAX_ACCELERATION_M_S2 = 0.150
TRANSLATION_LEAD_PROFILE_ACCELERATION_M_S2 = 0.149
TRANSLATION_ROUTE_HARD_MAX_SPEED_M_S = 0.040
TRANSLATION_ROUTE_HARD_MAX_ACCELERATION_M_S2 = 0.150
TRANSLATION_ROUTE_PROFILE_ACCELERATION_M_S2 = 0.149

# Each tuple describes the C2-continuous strokes in one fixed path.  Sharp
# square and N corners are reached at zero speed.  Circle, S, and U are
# tangent-continuous and therefore use one stroke.  The 21 fixed decoded
# anchors remain observation descriptors, not timing knots.
TRANSLATION_ROUTE_STROKE_TICKS = (
    (450, 450, 450, 450),
    (TRANSLATION_PATH_ROUTE_SAMPLES,),
    (TRANSLATION_PATH_ROUTE_SAMPLES,),
    (TRANSLATION_PATH_ROUTE_SAMPLES,),
    (555, 690, 555),
)
TRANSLATION_ROUTE_STROKE_LENGTHS_M = (
    (TRANSLATION_SQUARE_SIDE_M,) * 4,
    (TRANSLATION_PATH_LENGTH_M_BY_SHAPE[1],),
    (TRANSLATION_LETTER_PATH_LENGTH_M,),
    (TRANSLATION_LETTER_PATH_LENGTH_M,),
    (
        TRANSLATION_LETTER_PATH_LENGTH_M * 4.0 / 13.0,
        TRANSLATION_LETTER_PATH_LENGTH_M * 5.0 / 13.0,
        TRANSLATION_LETTER_PATH_LENGTH_M * 4.0 / 13.0,
    ),
)


def _quantize_translation_schedule_value(value: float) -> float:
    """Project a finite scalar onto the canonical 1e-12 schedule grid."""

    if not math.isfinite(value):
        raise ValueError("translation schedule values must be finite")
    scaled = round(value * TRANSLATION_SCHEDULE_QUANTIZATION_SCALE)
    return float(scaled / TRANSLATION_SCHEDULE_QUANTIZATION_SCALE)


# The canonical geometry and schedules are generated offline by the diagnostic
# reference builders in the tests.  Production import only decodes this fixed
# integer grid, so libm implementations cannot change either runtime targets or
# the algorithm signature.  Every progress row and every XY axis is delta
# encoded independently; peak pairs are stored as absolute values.
TRANSLATION_CANONICAL_BLOB_ENCODING = "base85_zlib_be_i64_delta_v1"
TRANSLATION_CANONICAL_BLOB_VERSION = 1
TRANSLATION_CANONICAL_BLOB_SHA256 = (
    "c3de5f0eb864d63059b009313f56236ff720df9247b05d50391660f53f584ddb"
)
TRANSLATION_CANONICAL_COMPRESSED_SHA256 = (
    "e315bafef7df586aaba0a85fa6aac8e170f814628263629c92f80d3503bf475b"
)
TRANSLATION_CANONICAL_RAW_SHA256 = (
    "eb7b18ffe69511b99a6de8d0faf854c2ecd7c3082639f3424e5c3d8955e1601e"
)
TRANSLATION_ROUTE_PROGRESS_TABLE_SHA256 = (
    "6865f79543c5a42ab9019965538a8a4f0aa0d35f2286c378621ab48d41e09784"
)
TRANSLATION_LEAD_PROGRESS_TABLE_SHA256 = (
    "43ecd7d772e4db8dd1fb05c47ba5e5d9c659310a705dab6b6beeaac0880664dd"
)
TRANSLATION_PATH_WAYPOINTS_XY_SHA256 = (
    "a3573b55b5d8b233aaa746455f6099e3eb95b7e5c9aefc3427ded5638c24554c"
)
TRANSLATION_ROUTE_TARGETS_XY_SHA256 = (
    "30aae552dce3d3308a9fcfe572e8bd0e9592914abc8ed475f8482b1892999d29"
)
TRANSLATION_LEAD_OFFSETS_XY_SHA256 = (
    "26efb26000be1dd9e786e05b2bb6ece3059b3c0e6a4511fad52a66b194fc1498"
)
TRANSLATION_ROUTE_KINEMATIC_PEAKS_SHA256 = (
    "c872342019ee1a06a3bcd8cc9d0e44865c91f286d9b099ff448482d1836cc7f0"
)
TRANSLATION_LEAD_KINEMATIC_PEAKS_SHA256 = (
    "ea5216203faeb632e8661ad40819b787c1e847faeb55e92d1e6e3a905d6839c9"
)

_TRANSLATION_CANONICAL_BLOB_MAGIC = b"G19CANON"
_TRANSLATION_CANONICAL_HEADER_FORMAT = ">8sIQ8I"
_TRANSLATION_CANONICAL_BLOB_B85_LENGTH = 87_783
_TRANSLATION_CANONICAL_COMPRESSED_BYTES = 70_226
_TRANSLATION_CANONICAL_RAW_BYTES = 254_132
_TRANSLATION_CANONICAL_SECTION_COUNTS = (
    5 * (TRANSLATION_PATH_ROUTE_SAMPLES + 1),
    5 * (TRANSLATION_PATH_LEAD_IN_TICKS + 1),
    5 * TRANSLATION_PATH_WAYPOINTS * 2,
    5 * (TRANSLATION_PATH_ROUTE_SAMPLES + 1) * 2,
    5 * (TRANSLATION_PATH_LEAD_IN_TICKS + 1) * 2,
    5 * 2,
    5 * 2,
)
_TRANSLATION_CANONICAL_BLOB_B85 = "".join((
'c-ri}1$0%{+BJMm+}%k+LX0HDNJ8`kDaGBblmf*m6m4-R6ewO?OQC3SDNx+CMT$#Hu>_5i5dAoFt+{W)y}fPkd;jr`|NZtE<BoYtPR`kTtw-kbtoiKJjY@d0W|O8(G#ZWe?f=rQuCaRyeqkDo@$YZ{&lQcPP6v&~^4Qz|Zz29SO8<s3uRa$625G+lP4AfkZBt$X-A!zP0U?IK=vCc-wR_wFwjJOB?6>1CaAI&OaQOxCIs4|g0dEX^1I+8TNDo@~F3*9wE*ZdX9d`hy)vpEI?Ryw_|6UVZp3M*+pzhZJzyVHUfU9hl0k5B|s7FKV$lk#B>pcdJH8lbL*sT#TuT6D58pYlm2psxg1aMD(V_?C+LwYn$dGY~p;;D7O3l_cfXyUYS2(a@vn}Hi16a!!PZ;j=v9k&xWHwo8k`f%X)dNhl<{0VS!n;F2{Z{qc69<k#);G|Jbz`ISZ>d_*`KMpuOcRBFMf&O~5tU0IxaJgM2;G5azdbDcuyc2MDg}r*TcJjdbSr2~T1-#mKiXLrZw;cm6{C+O5p!H>J_Xiu=0gtZ4e%l2`rvayRz6Z?o$kwBM+pf57`$OR;_2>{V;z!_&8nuDBpJRU=JNx7UPe-oOqf_Ln*T5x%>gmzhXff=`d4T=bz+2Y`>(Qn5gFxW6h$KC_R@hqyIC=O}V0JIqtJ{Z@u&!?B&nCl6<HPF$zuWq)9zE>qVSPMCXTh#Ko^8bYDzxo=0C=KGlpZ}JUu*y_Uz?yuFN;<%Q?Fr}m4OdGd#Fe6rcYpZ-UnjgKNTx?f!$Y}*Y$xOeKhe{N1wj;ngMS#yroCqT4P@THxG5xV<ophd|ssqvkQSQwqhOp+Kj^e_#MvOrbqu^=NrH|H(vn@5^>)FUE^Sv0cX!m&|_d^Mm6BV?}h`5A2!!xP`CBhfoC#(^;kJ<FMhA`oPs^T++BV27~Jl6tVi(vG4Rh2uPL7a#}vS?LVml9?+vYCF&X&Hw>5wzUw^5`DxGe_->dvM9QP6C+HEv&@Ny&I',
'MW2>>3=e)53j89kBk*4BAM_Xzdl7ycF=Osn;G>ym^%$9W5bF~;+uIZPXcP7?D*ofofz#^19;5CxzOBcq;lq0Z#~cy={d3Sbge%YWn}B_4!XKi4$iRBU7;JwZ*m%Mk;KGmay)pM2)zo8b<?8tN*Z~Q!=h)qG=YR!q*pF%{ai@XfW8q)bj)&pD<BTeP1FUT}8#v*~RN#?4SiiX9&!_7#J|Sd1aPS%UN&My(@PqirCvm+AUOu?bgq9z{-x8*LSrvF-2K+4{eOMJeCi?2qfz2Nd0*?PU61ep!J}2>R9{eiF+!)uH6#cXT@Pk!rfnP?zev>xOItu*dDB@*O{xR%tvgcgbd2;orjleF;;aADyu8snJa~;={d~g-~Gx>JBH!ycT>_}((Bi^s8yasVxSECL5OxNZD>_OMh1lO$_XVe$?^-0*V?z{TPAG%#LX917T$NK0lf3OsI?@0>qX$0b-E<5ffut)>HO);E|{YWwYc_h&0ssqqrQFowo1;o1)mrmEpA^)B4<L~iv<md8xl<(zr@Va=Nyl&nH?~C`z`{p`uUARtMH@S{nS7n{K?%W6N3-^ipCijv1s_ZlOopwOGpq)s&p&co9MLVP2Njs!nDt1b{r5#JVrkyKxPd|`;K|fLahJHl9l72?NQ~Xf+CH++KTl%r|Yx=q3_lyHFE-+3gaYM!t#uX*b$hgBeq{Jl|rx>@CI40v7<D3%rWE^B%RN|zJn~bAMT$OQ_ahGwJaaoDeGHx@DD{)=MdB%O_0p<lIPsqH%Ji@%fJfq|tnTME{n5USxlsqQ$8uJ|U9`m4*7iFGg-eewSURCm}%)89P%*)Kv%-hW4N?w<Fo_U`<fV_Y_fxJQC5t3JsXOMT0hme<$rzpHd@)+_O@*MIWg$GGqM4m+6L>@(6MV_VbF3H2l%gEEn+Y}xrc^!Ekc^`S8!V4u&ByUuBq~w+4ndF`1p$adRJe9mv;jxm}lIJSCSMp%;VudG5-b@~?@M_7k$-5OEPF^l~I(fUo<0Y>r',
'&sTUq>i}67uuf3w2G$X*D`cI)x<jc$SeLL)k#!5}7^SXZox{3^b&#x!SSKlU6YD6}RjjjEcd-s*UB)_%bsOtA)^)7&Sog6GRO>;t9#rc=wH{RKLA4%K>p`_1RO>;t9#rc=wH{RKLA4%K>p`_1RO>;t9#rc=wH{RKLA4%K>p`_1RO>;t9#rc=wH{RKLA4%K>p`_1RO>;t9#rc=wH{RKLA4%K>p`_1RO`Y2<@F$6Cffyu|15CK6M?Vce;VzAN#gfEI0ZCpguJ6M>N*i<{LOyxJx1c!%w|OcEoz1Vt=z8wZQP20b}3&09lnA+X`G6h0$t}JPHQ|`*b8DF3p>*Iq#OtO^%wUUc(emBBoX$i3A>57uZdb(3XGkI{nI4O#`<e?$HafvauwHEXAbN^(=Y;YM$<G4{;6qo6Mm>^{~!kVp)K-@rhAv)fxWMF0S@Tz4;)(A3OLd@A2{Abynk{C>{K&jAmX!T?k(84W=VJP`QO@${a*LGxUMZ%5EnGNA0GrB^nhP$PJ9Zx)0}@Q*6Wuq#6I599RWUcp9g&I0{)`OjuG>rg@cA++YB!wUTdwIA&zLBGA;nUw*3kWoMR1)n1gt(jo+RKteH^&Y}l+Vu+_zIVCVi_fxSZD&)Pwz;=e~3z|OQ^dWrYXZi76hU9tsnP`kz-`CYqpAM8T=V?Wrh_M{H+Lwl);`20Hwh$Gr3-F^aQZW2E)wl9JCnaziv8aTxwpBhxmM&30D{TcgZ5O)&xYVh9e`oLzE9|JqJ$GRBwIvxxhS|9P(VEhYwuffcX@FRnzlaOZ(){o2q?w;ufJhuNC@RBj|v%$UI*k^-hPmvc4@+U5aTbLvwo*Ft?A`Tk*=D|J;BZ`qf4O6_mg!{B_0qn2^{%F|SdpvOXdhkiZFB`%x4d<I99vZH`558i!<1*~T@aP@bvEilC&cOT0@EgPQuMuYqOSQ;TMmAHC-;BJ2kq?c+9_oO)-2uSH--tN)!7ALRQU3#ot45>K5uc1^CSqNTRxC$8GTQ1E',
'1U$SE{%UloCGxk?@6NCzqs*7ES7U>_$V0}C53%pY0a|xpwOD+gaouqXfbAYM2KMRj1#sk3tdH@G8HFOO)HeWb_izOs*TQa$uNLF_j8p9pe~sV7BcGU94ugN0c%4Q3HHnN`2&}c$4A{CR_>oD^G+dv_h&8ZBlNrN8fGazK2bk<^4L>nC)gAU`a_dX@x5<lxh`**9GsH<#$DW9rrh&Jy-=>KoPM9_>fWMi3v>x|wI<z<Z*L125`)#^BpdoO3McAw9$uQ(8(_4)ZuS{Q#!+x1*Pl<k#i#P1wEMzMD-mJO_c$rzt1@I5EUU4FSj(UdcG@G*xeBNyBWbjq9{R45IW|#UPUYI=|hJ0mKFc<vW-1Zpk!8|~F9+=ny>uTO~7wp3P<3QNC`G^g;fAiV8eZVz>Cz$VF0KR8_xf9~Hd1_QYV6n?1qya}O+@D3TJ+9xPdLZ6!(Yhgch(+IVxL%9#C*A`ta$O1BG6MYE;=~K=yTzUUNH7+eMc`SMW=ppKE7nKcvaDtd|FCRy34UbRZ3FnJ<%l`B56d}|ErIK&fuCC*UM}8u{a}0G%jd9bE8`#wpyxoWlU4L-#8<0^p_hPNm&4z!MtH(6t>&!9x>#+fg}h*OBn@%V>gH0^4^|l+kvFW(f)FRIeYCg^>$oSYflY5N0rtEOzHB}AclfjQ!aUS#*4sRAf7WN}!Je(}Pe7iq&Oamgo4ud|Ho?OXH*9MDhWoPV&;mTrX3%BuZ<}cyv930&Gv)vf%!b|CTuVaSwRxEVKeRR3HVs&DB>vqtzL~)05m;Z_KJMUuwqIBw4%#lWz;)Q}c4!0q*$+I{HnlqT-A>c}Q=t1?e2!h!^Vny*Mpjr)yN}zV-m)9>J^asZks0p8Zu<c6DZ6vOg7?}zYJ>P?S9~3xZ|~X#-)A40g`e9um=1rj?;e5gvmbpMan*j|46LL5_WIa&`}0=Fi}sIhz~AgkHo^}b+$Le29IEz0oOfu{A{N-=J@8M5amlz(hb76de}`SQVebw<x4^nN',
'r1i`LZ#S3-p6%$l4u0ua?K14pvAF@h-?483;*{egv0skg{3!aTKN=%%IbLmzIN|te6Z#NN=H{rgo&5SD?l|eLf~PvQtA##;)1V(<XHGLB5$BxNZABl$=~y)Qk<;CS$Tv=TjZkMg+uuiAa<1|@*5A3V8}g&`NBeQV&ZD~F{+$=vqRw^ReiV7q`NDAUO6Sxl#CaEO2I7W`=T7iEm$(t|dzTjPBaXQAcLuL;nVg1r@3QJJ;+D(7CBFf0evZ1{CA%Ydp{q?z#0%HpP{du=+OFWkt{<8rFS&kR7$az6CgQ2<)@R_)uIEx=@2-!ZVBg&|X|QXz3NK+lZq;(Y^WB=4Am6z4w^#w3;*PxRwmJlU?{-KBKI?X?1@fL-P9I!{yX_a?q3$6o;7{)F?}s0|ce#za*?nXo@}m1fFYpWZ9W~HbasTOK?7w^3H28^!;ZD@)9^Q9wpB{<E@JEl<k>J4|13Q7=c}$yv{qb148}Y#7=tJ;lk9!VSUyuBnGf}rXev0+45dJOdk_z=NBTiNLSc}iEFs>^0zrxaPSf2_%%*_H`K7qbUg%`Pqr=Dg3s8c=tT4TLDQzqiNJlpRS{qi9X@I9Wh?U0W>zt>^EJx}$<=Xm};ANk$0=rGp9%k?P=Zm(#2#6_<r3HW}m-XHb=PMnDN=Jm}w@EETj&%!^wu00d~o@tJ{-P<Y<anw7gCf41%R!8iQ_XmUFKi;2Bg`IoPU554X-m(RCp!eB>sF%DSop%P7UPHgCqQ|d@!xdv6fu~k%mJ0u`*f$ORrix!aL7uDl?eE|R6%X77|EhT7k`|bC47}aPdI#3Yr}7Heq0f7h@i{)7`y)>HeBKOp<})t>@yBPYCE~Zw*@w7)pGSM}`97snkpFx=T8ndo*ox?%`8IuoJnh?O4gA7)Qa8j=-<6)=VZJ|JM*Q);HbJ~EBLe-wN)|W3w<`sV!0%P6;Q_y`)L|F+a;2dSQ1?`tbqD;u(#Bq}r%ER>U<Z|c8?+VpCIjE+=hO>4%rE>d;*ekc',
'2B@?Ay6qHwoiVQ1f4@aTvEF_=e#ZCs{Zt9|;+HxA{^_sXi+t)|p#Xm9UoFlC*sMR|oPVDc$m9MK&%$5)SG+=<=fBSy`NIE~0Pqz5mvM*(0jBTax&nObW4{8D>LGpyw5|zWA21*q_Zcv?BI0SlY9rWFz<~!?pMdN8<^wanLO&qTq9u4?puY>^XQ1vfcxqtVNr<<BpH_w61b%fB{olavhQlrb580s45O`w+?kg}e5P3PsVjb2i$lnKjf*{@3$P+<r3b8&x13%aeoVo{bA!xNZ_CM%AEBH~+wFT%e1--fi{!`h^0M}pHHx%|#Ik7R~YUP$akoPM08-{pPdD2+$oXRW4pub&t&rsya%9pyM9<2Pd0X{F-AP9Lg*rO1<BRE==t-+0^qOK4AxEA7V@aSy(Uhsk?xbNVt3FyxSpZ*!w8GOGz>d4^yyXen_*th-&7<>}z8d571c_gI6WYn`EgKp!CLZ<n{e?nGwI|1Ck82cFV%SqTl$n#WuU#O7={u}CPgLo1e;{ZPmZDa+$AKI-5JS=qN1Ms)dxd(CIq2JF!A0+g+=yQeMaRC1f%{qbnTg9R;_-&O+X0VSc2}{6hsx<e-K2+(o0`^vAoE`kV%EIB`1y#2E0)AiRQ~=^_m3w^=x2xo=1z!rYypDVpR;duz8J6gb{1MjN4}G7oUV*Upu(95_?yv<Ghy!7pp5uPQjvqz8EbR6)^aa8)>*K$}O$%XH;oht8J>juB<kRqm7tqHE@7he9w|#aR_7*-X+6B1gYxq<6{>S*9@GD`6!{KSYP&bB`EClb4aN38TM}%BL9*n4Q8~Yj2<_7BCh<;~bzY!BQqwg58XcE>rVoP)IiHH++;AIiFj=?V?(tDzB7HO<Oe2Vm#4jvsDX^HcX$oEIX&LTTJz<opxOa%XnoHPl35xL|zcyZ*`9P$2>Zpb^4x5KgCk?C>B?@>n4$hT4MzUZ?=g&T<T!dll*zelxQiaH~zUn^W^)VR_J;QVFC15q18(MOItv>ACW>K9*J',
'M^x%$*mYFlefU{byAb5_s{TEZ=c~rg$NQ@``T>2Ts$I??4ptp<0l!yu%5m6V)nyw|k5%0^5q4emL^IS&Rc~4&A5?w5AMvbeX(#ZrXopwm6GaDpiZ~yg@C1Fg=*Eo@Z=<_>hy9Em{1SX5dU6oreDso*;{BTkf@ek_8H06?{$(`!DA7+o5$79knj&AtSotD;#&|!*`o=^ph8@M!t`577X>$#B5YxL8@?^}&tKgq8Gn3)RF)J6q|6_LCM|_SsDf(40H|n6@5%csT<b#;PPf+K^TKC2IQEbI_u;19IYUq!}zGnp99NX$RelNDir?8vY&+JiO#7<p_^NHA{KFGhZn`WW^7<(`sc`Ei|b@*lM{juWTGq++ts~KEGAF!I^eei>7{`V2Ls>NPJ-m6w;E9#7DZO4NDRqK_4yj|_Hm*{I%n>rnJeYGVO;4jrSF2=c6wf&~(t5iGRUA+I!cJ!aCJ%5EfTdl|gyeQ5p7V#|3vj+Afu1X@}eq6FY@@-thV&u2Djwi+WSKrZCpSY2Jr~~4@+J|@^x1>6*J8r{H*nQkS7x1LGGrhsr;%;md|NZ0+_(oiA0sJ!F*c|s2?_`3w5nm}A`8Gc4XXKUm8jH{;if`H+eU<nR@^L@${bqsR$B%SE{*IqI8TmVY;WPM8{Msb=ar~~4u;ch+8{kLrmrlX2;(xsidyRi_9`QK7U?<{qg4tJiUxG_h_-TT_A+9GO>bp!}^;q;@5*qEqekOGAK_4!m=SbLV!jKb);|b%7U{49Ne8ulAi^V=BY>0z@Cj1bD^-MT!jQdWwcp3aN;qG+UPr}m})aMDg=Ma|@4Vs{iO0+qGyqH+Q2mO@9p#G>&5~DW^1y;WT{+ZbD8SE*sZ93K|vD@$9Lx}^9fwv}(6!moC<VLWQ#IJKv2PLkUj(%0*M*CgB-4n&{A9)D>O}r2#zUM|i@Ylo#^ANuhUv3n?m%jyhHOcTB)XhmYqfz%Ix!1?{CixrSyf7(zJ^Cw23CXa#r1uUY4kR@R1%F9u',
'H(C65*Go7LOzLlf{FXF47<n&gd?LO-X<9t^M$&vg#NDJ72p5{9^+(|ENjnB1J|rEmM?WIz<Z|%qq>HY|8%Z}ugP$h-ei?Bm>A5}lRZ>n0>XM}5_K2&=CSAZUl5JbTUy|Ko;djZthUlv#2Oq-qBv<VLdrM9%z&U&Jdn1vTk{dn)&q!`n8|Pffou*@5lDi*49iH6pfzUGyNe2%~9`y|NnLP0_;#KnWHRAi`^}zloFLMMhNM5}eaWDD%XvD?j9cy6^$v>K){zyLB0``@BW*XvG^2KfNyX5P~5a*Nc9Yud9`O){{x}JXlo|c?d7w0<3`FZgF<kERUFJa<^b<<gW4S&-)<Y3=*?(f0xbrr|JKXd_WaPFZC-HY=XUDRIiQ(fE|e4kD?8vH}|UM*awu3k3kF<p~6;1Rl36~u8!`#Grhb)B<OFY3D2L>|)h9)*0N`(!ojT{n0S@}2IpAK+KI(W}54bQ4B^N9!i1d<dNO3VyDeJq^4=H{TKKs#`oo=p2?mMLf{0s)l+_w{`&XfNtYF#9Q5#wa9zA9jnFl{xAdfrTeifc&P4>KkQm}^a|`vcd{SigYHZg>_c~<JMzBn;xX_p-7i+)54!8Mgnr{zci6w~?qKj^-EaM|zq$u4QAg_@2Z;BjJ`(HmY^GSRmm*&4($^y2>av^=$8|ZMfLG{V??&Fz<)?~$EHp;^*A?3$ALvSr5pPm7spyZV80<pdGR5c<@SGGAC-BY`)3v@rcjAY-CB<R}cxj5|1Dt!MSo;SHeM)oOcZzKv)Xgb&1CTFM?7I{S9g9w&qlx%;$Nlib6sJx@1wQo@x|feO3OrRw=wV#O3v6*#;D`c&v+T<w|DErXzsJwv=kj~_y}S-y7q64o&HLbe@jiLqTnDZT*NN-Kb>zBoo#nc7AC!IJK5^f;kK9-8Gxwc#K)axw&~9i)v@6;f?T&UxyQH1cZfVD~YuY*Oo_;{Tpr6oh=tuM``WgL>en`KhpVDvX$MkFZIsKk-fN_Cwf^ma!gmHy&',
'hH-~+h;fN=igAl^jB$-|j&YB1ka3Z5QpQcjQ6;W2&NA*Y4l^z@PBU&Zjx(+^&NJ>a4=^t<PcUyVk1($=&oJ*W4>2z>Pcd&Xk1?+?&oS>Y4>B(@Pcm=HJj%SP<XPrj=3(Y#=4s|_=5gkA=6U9Q@&NJz@&xh*@(A(@@(l70@(}V8@)YtG@)+_O$#cki6dpugM4m+6L>@(6MV>|8MIJ_8MxI9AMjl6AN1jLCM;=ICNS;XED0w7#rNT4GJIO=IOUYBoTghX|YsquTd&z^zi^-G8o5`cetI4w^?<NmdcsY4Gc{_PLc|CbPc|Yp_)&;B+SU0eaU|qpFgLMb%5LuV7PEqO>)-kMWSm&_rVI9P}h;<U{Ce~4`t5|2T?qVG#>oV49O5Mgfj&&XDJl1`z16dccPGsH4I+ArI>rB?2tV3m8$~sl4TUp1lu4SFex|ek@>tfc)teaU!v#w^HE$eR9;YwZ3I-PYp>v-1ntn*p-vk$<&0Q&^&8?cWc`wHwcD18U^A=sB-pMrf0_A%JkV4s715B5RW7m<Au_Dz&N3i~STv#{^NJ`DRZ?9;Gs!#)oCI_&evz7P9AN?(Y5BKD2gM`B-zeJ1vu*oR_YihU~ftz;jIeJ!QW#l9E&VC;*rPsY9(`)KT|vCqc7o9x4}FQ@eB*tcUJk9|G%`PlbkACP@P_6cR*kbOj@ugE?l`;P2GvM<R#CHt1_W3sQwKBw$^vJa~CMcF51-;{k+_Ep(uW#5&3SoURQpO$@FrH{+LF8jRf`?3$rzA*d5>>IO>Ec?prGb?>(_MzFAW}ljUYxc3(*JhuaeQ((ZXJ1_Dle2HmK05pA?6b4)&OSW*^0H6QzP-}NXJ4Ove)j!&4#0B(o)hrgfaeJET!H5d%DDs2A$Tsqa|)hY@En8Z8a(Ha=N>!<QO-qpPQr5&o}=(wh370hci}mVJeT1)jdE_oa~z)Q@SKO|K0F8Fxe(8Z<hc>gk(6^Ko-^^>iRVx}m*P1U&#iclCC{~Z&ZV4t@f?ij',
'Vmv3~xf##Vc&^5CHhJ#Gb2#N(j^}hdx8pe;&-HlD$8$e<4#;ys<(!b`hCD~)xgyUQdG5$_NS;f|b4s3DD(9Fy*W@`T&pmk#%5zbklge{bo}((~syt`qxhv0Mc`nOyTAthT99N#}@|;&W_vJY-&xLtT%yVO&BlBEYo-^~@SviO1xirtId2Y>fY@TcLoSWy~@*JG!;>tNW&&_#`&U1C1v-8}Y=kW4ep6B$+xjoPEd9KfMexCbN2S8l_bpld1KplaiE1=GRx&!JEs7s(ufw~2$W1y}<(K%4}Kpg~i5!6XgH$fc*brn))LEVL-!=NsMIt}VJsN<lngE|kX`=Aa)(S=YaLfr^;B-E8qXF}ZxbtqDoLY<1DTcM7Hx)$nOsC%IfhPoK3lc8=#(a}&>L!AwEH`L)!mqVQnbvshWLtT%e^P%pCIw0zTs1u@Yh&m#vE27Rw(H&8TL|qbfO4Kb;$3$Hdbxu<EL>-i(i=s}7x+&_YsH>vRin=T6u%s@FIxR)FMI9G)UDSC|_eC8TbzxE`M%|dABcra2Iy36ds6(SJjXE{z)})S&x;91UM%^2AaMZ<7Cr8~Jb#&C#QD-N0chuo2x;*OisN18CkGekU{HXh*4p8a>sS{LmgVYgHS4f>9b%)d;QkO`bqSP%?$EfHUsdJ?6kvd4~BB_(4Zjw4m>MEtqlDbPphe=%~b(+*|QpZVMCv~3GeNqQ1b)nRWD!NhXNU1BO&Xl@S>QJdmrB0Q)RjFg8u2s>wQuj(7EOoKe$x=5<9W8aW)Y(eiEp@nxE|)r8>UOE)rLLDcU+R9T1EwyRI$^0BrjA(A6;o$S-7$5@)Fo4=Ox-ed%+xhg=S<zR)In1ht>~nwo2HJMx@zjIsk^2Qo4Rc3w5i*sj$7)wsq<EJ-_(Iq7fzixb>q~LQ&&!%Id$jMp;MPGb?VfuD>`=S+NpD=?wvY#>f))Br*57)dg|(_v#0K!I((_ir%qqd?Ni53T|agH)ctb~fO7$y6X4tc=Lk4gz&QiX',
'9mqKZ&Lt>w3Y=Tu90TVXIOo8*2hKroE`oCsoSWbr1?MU_XTiA(&S7vaL(XY%ZbO;l;9LjiJUI8kIS|f;a887CBb+1QTnXn)ICsK16wak^PK9$ToMYi!i=1=e+>0^?!?_sF$#8Clb2OZ*;hYWUZa9a-xg5^vaBhclJe=#{oKLHZ5poWwO}V)s&IxgDh;u}oE8?6H=Z@qY66cbXIVH|5agK>|O`LP$+!N=ZI2Xk^Db7uCj*4?toU`KG73Z)xm&G|P&TVmyi*sF^^WxkW=fF4@#yK(0jd6~Qb7h<}<J=kN&^VXIIW^9$agL31ZJcxC+#BcMI2Xq`InK>-j*fG6oU`NH9p~^km&Z9h&h2rIk8^#T^W)qf=Kz(tK+XwrZcv>gROblQIYM=gP@N-G=Lpp~LUoQ%og-A|2-P`4b&gP-BUI-I)j2|Sj!>N=ROblQIYM=gP@N-G=Lpp~LUoQ%og-A|2-P`4b&gP-BUI-I)j2|Sj!>N=ROblQIYM=gP@N-G=Lpp~LUoQ%og-A|2-P`4b&gP-BUI-I)j2|Sj!>N=ROblQIYM=gP@N-G=Lpp~LUoQ%og-A|2-P`4b&gP-BUI-I)j2|Sj!>N=ROblQIYM=gP@N-G=Lpp~LUoQ%og-A|2-P`4b&gP-BUI-I)j2|Sj!>N=ROblQIYM=gP@N-G=Lpp~LUoQ%og-A|2-P`4b&gP-BUI-I)j2|Sj!>N=ROblQIYM=gP@N-G=Lpp~LUoQ%og-A|2-P`4b&gP-BUI-I)j7id|IQKC@fN@Lw6Xa7&<W!E&0@v%jSvTCZezhOHI+pFNfQ>FC)VjS@`|SJWAH#t+cBtTHN9qR75il?em+ZsxT^WC1Ne$&zx!xmH;LevniqTU-&)fdsI#<wdv*)^t%`b1`(eKDpK->*PQPt~IHf(FD*P#R2kgbbYS%Gf)u#eGG)5jXm~aR2++fpeT))AMp0Ec)!%oOkhLQcSABLU320t*IaSr=uc*tJF!St?(%SOKYu+K)V9l`I6',
'ruG&8edrMU)hO2#{W{|+wb8FJ?mhs0M&spQf-f82m?-j!Ll49&lO{pP2PR*gK>xzzOd|Zo)MO&!i|PB{z+OxzO-BFB^i(|Jftl%6?7Lb0G~_L_>A#8meCccKo4HdVuHU?a8~P~b-#kXWVgBSp)U_62<Kfp9L+hg+v^cl}b%>?OkEnMon|%QLv0So9@Sn#$P={GXo<x4J8nqq!YIQaUdDYs*ANy$CZ8i4KdiyF|myN~?@yDjA1L^^r6{Fy9Hm}+vf7+(3L7cIj^#ShN_Wl?T!HbRH_jX^rfc@EBkHP)eSH6jQ*nZ?y?7#iRAi?h|T>yV`7<L4F$>F>f_vKh|8se4XkdEN9j%Np8J)FFcqwaGW(h8sBbj}C7+u6qte&9T;!(>rc9fy6n_;rAvyNvKcf86D=8ThMfWe?b=>)0mPZ`bRq5TD(`ozORPoAfRCjoZEV;jiv7CAfe0>F2Od?vK{MzC4nrgXeq98v|bG@!~VAXNB6s;7=8n4TGOn$Qg?3_iXej;+N-ly#|T?KpVtSuT}|&PhOjBvEJT>*Kobw9j74fd+!WPK)=Ic6YRcXS9d4iJ`t{c>=q*q`Sjd?_4YY5A3VX=xg*XceEa5${?4&J;E$EuwxSNN^vTIM(O+7R{Oae?4*AD#z-8nmzmxWe|Nb7X;3@u}{MHM2yqD;+xo$;W5YTrg_*uZ=fyirt4o|>O1AEvbPX+GH1g{FR9FI5~^uaOYfuOBh(T}dIt%o>Wxy3Z}^(wC(ihNo*#|U;ETssbWhv0dZu!rD>V^RNvRGkez5;9(g{Rz1^5pg5bt1I|ZXrCKchtNIuQQuZE8j1C&(roPyz-3*~KdF+s9`+knRn+TYqpl-Ag`L`mI3I3T3GpGky&ZT^__yOwZ-qbo6nQct@&V#p#PCaqw-JXD(SMFK@<x3gS$`zrMdZw;IERkBvKV|L%B3ClHLCqo^y{ORg`yuE^=mTXXw`}b#)<Qj4M-$azttcgRK0%;b{}2wg*gA|JQ~*%y=(~b',
'TlB4qIG2ranSuBa(_-^(ao!Y$x+mtWHTXuXVFT1RvB?kb1Bc!O?}^<Kgm@kM_#)Q5n&(B>VYQY4cz?AimtaTL4qgJktd><toEHY12H%TocM$gzH_ZU{5_ez$&UNEneg^-E_gag%7T?%MoNJDJ4Bi>PAscoWf1?5HA;It|{3Rjc;&yQ^TZqp~nA8uvFJY@M)+^zrJL;W8P0QEf{8(I9Vv`xL`@|t%!LJe*pMrfR9;%5xbmH#?=x-(&rHk|WAYXBgUvDPXDXDiH>?LWc`*v|&AB%jQbm}YYL((HR#Ia<}J*-Qz=Pm44a=aDzadL~H;^+Mwh0bB(lj}m)AoL)~yFzf^$>$b>w<O<B2LDaYwMYL^XYPjnqRz9q(5-~;7rKlZZE-)k=6<+t-A6v?x9A2oMqJa4*?@CQ-Sk+YGg*{@^MBoUw-G0GTeF0o<i|wtcir)=lY~yDrJz$+{e`aOj-Swx{N4<GKHZaz@F(5#s)#?jS5LqvbeU&`9wqxG>HuA?(>8%)0?Yk(_RSXM-v8>!>vEsR&;Pwy&2rz*>*4hs-#W3}_49ssf4tx6v32Eo^e(qPT(9PSKg#tSQEq*?-g5ol?nkiPpGxKSi~Gm@<o<HMTMVrt?V))&`;hiR`%&y^eeCIS_SQQtqMSVj`A#WouW#))s+>J9K6R~}y*HWBshmGZ|9I;!4a@n@q<)pl`4|07`d|DFyj|~~x23-hkp8=)tUuGg-}?J7>Hj@tJaCoq;cFQ$Udi}Tvs^r3e39{n@kfbAj8DlAf0yyAR@r!_#5cw}8UNnK!`(7IPL}a<yNsVxWjq}u<Lgi5;%&tP<7GVFCF8SGxp*z(_uF`0TrR#d-plyUe4ylq$OoQf^T*qKvb9`(k@<%C=WRX;kol=^*?gtsuV!oal+AB%^Id4U{3r9_+x*x|=F5XJe_oaObhgZ|zsh{Op=|zD@-g$X%-77{Z}Yje%<sp_=6faoJ3JXE`N1!eFN8||u(&Lrc*`$(m*pF8`Nx5Bd_?jS',
'@)d=@NIpY;qwpQcf8O$;k>&W2<VzLz^(vKo%DF7Rdds)s%keMC$H>pf*T~-#J}3FzTfVnk@;|S#e31N5;fs<#l25+nm(?WSJSh3+B*{k`l;x*y`Rc23{I#;^?~>mtd{^?{w|qE4^5ZWhUv47#^T2X^TJmf1ZH0eJK2CoAmaq3O%irJf`H*t_Uh;kN|F`wPV_6@xDqAlo^@FS@SYN!YHxA4CV?^0{<ZXR2Mb;~WW&PqP>zV!K>Kj?_u>Mi%A=XE-UV2+U-Iw)LV%hrYZN23p>#r5E9$Q|vK6_iQ#mM^YjjZRsEL-2bt@p;2tN&OJ%KGqaz1XW<{m6P!)|ad|mHLzQXqDPs%hs!J>(}?o)w8T`efM=QTmQbTha+Wuyh7H??aI~9vYuvrt<>ABzgds7K9}`6>-V?ye1xp;AC#;2S^u*i!2W>gjG?lBFsfWXf&GQ=I#|g5gK4>b1p5<h$JomLMZ<Fa4E8tJ?_mFf{Sfv?*e_xKg#8pv*lO|bDQiWart?_Qei8XuV{sJzsj)2;{UeuJqHpCjRrJRKE`d*IB3wF)>uM+ZNVVsR`)YOy{XoqJuS7qopSioZKPS{3nrYUeU$rzF=Rcaw$Hey@`5bvcbJZDqM3XiTeyAzVJ}LTuQQ!^Q&_>WlXy2=Wc%khi&Ly-%w^)e2q#y26`+aZhr}p$1#7*s^E{M|x2G-!?20mZl{RY(!;eHH0JOsNp7&i{~Xt2rv{%>%i7WT~`RfoQ-p+%PHGe>krJ~3=H4t`=dqBY{H;mSKWhc-N8mmusj3-Q3nZ7kN)sOE0)OrySY(9bkl;0T^!bTkg<F-9+n5D$&ryWo3`>vRFXH6C0DJ2PGtj=IqJQa<>QiKgvG;#{B}&b3UsoD=5`bMD|iOpbq!I@BchTktc}$}NyzO*?)C|1_Od7k*-Td@|ytX?{KUi&@wd<Snyqb&$u+7LO?vaUlx#VQ$(Hc4eMYfIhbQa2wQX=3D2A^Px0hFBZNp!Cx#o9z^`Jn4gb0WpQaK',
'`m>gnYY|s1>pH_fEhnT66z5=`;M10c+faX7#jk>0SPjj<zFX~FB+lV77lTh(M`VE4Sbwq{`g7~e%Mq8XU%bM7*i@Ny2H0;b{LN<5kKc&%#~R49w&8a0U)uqJSWnw+6A&kCvnt{Jb}@#i|LleZVjb=FPR2gj6$apb?RBN7`|ZaIJ-7Xd50Nh&On(P&a%i|1|L!nLo*VzN68X*1^(Fk#u~RR^6~}Mw;ZKf_A3?Y66!JU1*J+?B_`1_>p>uR9ID!7DbB*@kZO&iXA-+1FPsM$@*xwW9`E4`tIW8-FVV5qC`r>|ELw*8Jas9L&c%bW^oA|kF$tc*B+xtRy>^7|!{^WM$I^wRo`@!8p=dc6*;l6GM&QIN69YnnGh`o%yw8yCDI3MyjVI_1LmQ`Wr6<Tye9Idc;6868s{cW&Y&!F2lC-od)0K4`4ArSZJS^PfEGrekeMqKflHUK=z>&kHSp}pOQAYOWR>47-wy{;k7xxHWc!mlgFq~Y@_j$DB}Tk%*e^v!(CuE73%nl=W1_L;vE`|5L7EBZ>6;*h_6d$quQ_-<*5b@k1ON8GEFP>T0e8oLg8w$jN+#5X^Sg+lk#^e%Xh-@Fp!7r$GDLTBZD1-#S0+XURV|9T6YyZgWBjkpjHxg7o#Fk}sQP{7{t_`QIl2;{%O>RZ6u0w)x~egaQ<;CqA2jlo-k8XiQv37T00aU$r_OvKU34r`DvDz};lyREz+6n*c?Hx^<)gWb;J--Fv9244wY@)_<k_;xn<eMp4}#MzLJp~%l6OP(OUhTQ%T`x@##AMro5-DF%x=)xHAyU=S(VTV;54|EZF$9d2(Rhi`vK2zmFADk<MnG4-sSREJi2f`+Njyfjn&@9+>SV7weq3gT|{u@3pA9fu6{Y6}F_>)%n+z6k^s2d|X^gx{wF*g(Eei1(f!LK4sOkhWmDKimoBZnPE-4?lJ0eEfX6KB|0lve`ceN-!v2co_j03IB5WE}h<>UA{wDpf<LIEs2|w9pODcS8MK^=y67',
'$13*5x<^OOMjaX5Z6($#dO;)f^P<np1J8*r_)O@dLvxXDV?MBko-$^}LD*f)Arr*0n3t)TM~U_7jPu#pM*W1Id!#MwA$CJO*k$YuXY5ZkqhYweYEix6ht)pFz;##q$_)EhZTEKYvugLBW1r$If56Y<s@cM><37xXUBpct0A3fjV+eSB+#M6FL%g9s;!k|YMa1X$7RIpe_|Hxuzr}y!06l5^i8R#3@z3g_zD}@>MBR`Oy%Kyiq3vS)JYl3ecuv9!U*xTXBU{A#9~?uxNHpq%{YeZ!nW9Oo;|)HW*rN{kZQ`U%tY6|9AMokK<G=Wbc@I;>%f!+{IEPL0%sVFLNOpifCAEHuK1b508xW_Hrau;QFY8tze<dCL73Zi)cNUKmb1}b&xf$!}h&#yv$Kj{Ry3gi{d7SUz&&ho|in*cjQ{b1$3#%foCU0s7yGlM%Am(<ixr_Ouw4KO5$%SWd-#VN2q7UZtv5A;(dM^4@wVop{=vwwboX~aaDCWBcox}C$#_zZ+=ChpUiutW{)VI3t-^2N+Zm*r_2c2jm=C>|6RTurJ24WuT@f*}fx(pxGBf2+-2Z(vN>(OG4PRuu?IQGH!r+B1^IXmBb`23X04UkV$!ov`sQ=*05E+xJL>?tLAHTYjjjc$m~Deo;oKPjb7J=}jvz5cL|lm=$t0VxeDI0^i&UU`)N&iC>6_&NMseh<Hw*TL)Jb@IA-AG|N#C+}OX1J^}aC$1aUk?YEJ=DKqqxG&r%?i=@!`^tUhzS9n97t&5>H;Nt6u4repJK7=bl6FeFr5)3*rJd966+fU~&`;<$^dtHe{Y?5D{ZR2s`YHXEeoVinpVRMU9AI2f;soOc;|Svl<BW_uj6+IXVw_^!VjN>!lW~r5Pl<z!i;R<un=+0vt}1bsahGwJaaqP`#%(2zGp;kv%ec=xpyUPS3FZx%N0?WXJj1-hJS6iH^OTaen8%pcWS(Q*Q}Q75BJ-roo6Mt1US*zT-j#Wnd0ENR%-b@LGp{Roo_SyL',
'0P+HbCy+OgM@U{lo}us#@({^O$Ws*FLLMV|4S9~jd&q+%FCtG;coTV)<W=NZ3hyEhle~;PP2p|iagx`O=PA5T@<8%Jg(s3XN*+mGsqjqlPRT>bOBJ3<-YR)4d9A{8CGRB<R(LUavgFO=(F(67&z8KKJY3=BlBbimD?FaOUh;hMex(kObph)HrEXvyA?pg(8A{#3Iz-kbtW%V_Mb<H_Ym_>Nb&srrSQjaElB}CpM=5m`>nvG!u?|z}GS+FbZety%)OE7XW8J6JfvgK<oyfXTsUule$~u#Er&5QqE|qmE>sF<Xm31xaT&3=19W3i&*2zlU%sN`u)vU9Xx|?;ltjk%aD|I{Tcv;u8&R6Pw_5oyHfPDg`Z@@kR`wFtpz`ldhhhSeq_9@u6Q2H3`Yp~BD`yT9rD18z3No3!IeH5jy!afW8F0v29zKqhRVc&**9QJi&pND-Pr4Piu5c@>3Z^S;5(pO@iiG3&bp=4i*eJZ7I#Xc7MTI_SN?<M<S?29RVGWN~bM`K@&eKz*pWFL-wIi*j>z8(8`?CY`5$G#u?fU+;hKB3Y#WFL`zMfMrlcVr)ueM$By*|%gLlYLG0IobDQAC!Gj_DR_{WgnG&RrXogcV!>;zo&C+SgvpU&pJ2La((B2*13%;+qeD;I=2Vq`r`jRom;zdeSG%y|Dkj1{s%g@_utXEb^Q-@ZoU3Q=Qi#go!gh?&Pn9CNg18n@^a@a^4z72&MmR*In95dbK76`oJcu0lIKWebZ#Tcojd)x&aG9sbFDwtxrMx=bL;;f>f8pFJ-1WN@#MMQztOpA-qE?edPnEhpzJv#&mEO>$iJ?0^D2Ap`9ILP#lNF-b1r*M`!95EBmYY}w*&9!+<L#Gb6Z^Q+*+Pv^IZF1=-iH#I~V`!I=2nw&f(>`JkRNsb9;G?|1WfIq3`J2_WXg)t#3J9!5{0~w!Wit^Zc7Sx8gt0xjB{9RZwS9M(6fZSzQKo8j5a%Iu5Dp_zOC>TIF;je_!WTv#f4~Iu=FO',
'LY)hBFaNA_+fhz8LmiFO)%+WsTjh6jZVmrH=ho>Ro!hl{bZ$}a=-j5fqjQURN9T6>FY4Tym(w}@b)8$Kvbw5&qjMWkPN((HI=9~CbY6e1bDQ`lI=7mCpmTfuj?S&cpXl70zoT=@`vaX@jX%-3P5O&Ew>sr?fB&p=+gDCE_{Tc8%H?#3e_iJmT2A-)=Q_7j@95kj-_f~Een;n4^&Oqt)pvAmAN+yNt-(7wx8Q%Ta}&B#MTbgVDs`&=taBS$PUlM9tJJ|#7pv%G|Gv)cep#LFU(mVjDyIXcE?DY>sT)>w#Q)H_jVr4=R&>b!taB^*n>x1&@95mN{fW-azpM`XU+CO^ct__}<qvdjJ<I9BrB0l>aYaWib>)9U=Vnz_$Nn#LZUf5d;1ylG)XA68x%GNS=XUVFq;nhbuXS!>PC%I(_=nCd;eV)eTUKt);jio5V$04|aLz)RyZDFB?d3Z<H|?M3+``MvfylWK&WZd7I=6oB=-fj8MCbOX>>SKr(77EdH)rz?o!jJcb2)NOhjTmsfzIu>a&trf(78F6n=|6v(LZ!<vE}BJIJfi<o!i(y(Yf{h4|Q%Iy`yuh^9MS&Ztv*aqRP#IaW3q?r*jK0H^=tpI=3NZ=j8r{&dt2soZUa`+-%Ct>HXJqZW_)df>&#T{KSJKdrwDSS#u!*aaP-;8}?J1J_h-~V9QO>Z<uxtJj7_yHk=a}FMW*rHTmh3=<|7O11~n8V1WLmrJ*<0!D@MT#8;bUameF#9)<9G`(iO4?U?%v{K3WO0s2sG!98$Y9$kM$e(>D(gLn>%Lwoe;d?&{ukNMl|#W`Ex`ex|w2Dco5K2#OkFOeU@@0#L$QG0vA{$sxLguTVBSc>&cT-OnGknZ3$*lUg7zd*dMZGRu1SGP$o_-BI!ixB@BKYfLG(Y)?+tXHdTHt2`7^Sy<**m2oq<kb&-hU5ObZW7Ng>QVg}{I~aQ@qDQMqxvGBf9l^8yky9g+Te?yO%ij^qjXo0cg7a1gnxgre=_3J',
'<Z-Tej>)vTkC0DhIgAC5`TDmp@W%yPhM|sLGNu^owxZc4@Ss(G6|fI$avx(KH=Own_OyA)YviBp{XRoH*`0hA{i1!AQ;;tY+-!@v?IWvhVxLd+*$O^(CSoj}=XX9M9eL{FmbK`o{?e-v?*E3@RP=9d|7?c!{Ppvp$XgF8j)uKHK70Y6`?P~2{Pe{uEqFx6$hzn!=V%w>JTGtD2E?y|{4nsUlAgEq-=o!D?4|d6t#SBF{r+psCYk8@nAY;jG`+uQZC!5a<E_@A`9^*G(K=5!qQ8Hw+lHI^_@%9IYNx(#(0bo;(8qDD@BO`cKhgT%>#fi0+Mu7G>g!}}$gWa-+|Y*28m#XRY9l*a>g`_}?K4jwKee&fuj~Cv8$WKjo-b&Vy!PtXRjb>aqmS>}8d1Lb=V@tct+UbFq4s^-S^D^=t=ng)e*d-g_srAt2W`XG$3KSK1>L}XHEXa=pU<@|KD5UBTXp|k|NH=Ln@$>ioYc0f{l31w)^_lksXs5*c1+!;_dD$etJZiTEH#PN+l#i#v-ffTU56dk+o85w_C)=8m-gc>BlPEr+8zfd<GOoU%+uSyws+kr`f~tnpCL{4d`{bU_7(kkgSOwYQhlAH?Z4=lp5JIcnP{cYi`oI5&*}Rk+JTkRKLtMMr;lISPp8G|d4qP4U&<2T#?g8^)DHGZ!*vb*a`+|S9e=%_X@_{H;eLiRKhgp?cwGQ+;!@vo$baYi<nQrwl+Tsl!|&yF@Va=Nyl&nH?~C`z`{p`uUARtMH?AYsmFvuP=RR;>xKG?S?j!e=`^<f(9ndalC$t;d5$%e0M!TaO(k^MIv|HLS?V5H@yQd$}FX$)q8~PFbihf4FqaV^Q>8JEt`mywD`nlrwj021dj1!C-j3bOIj5CZoj6;k|j8lwTjAM*zjB|{8jDw7ejFXI;jH8UJjI)fpjKhq}jMI$UjN^>!jPs29%md5|%oEHT%p=Sz%rne8%tOpe%u~!;%wx=J%yZ0p%!AB}%#+NU%%jY!%(Kk9',
'%)`vf%+t)<%;U`K%=66q<N@Rb<O$>r<Pqc*<Qe20<RRoG<SFDW<T2zm<T>O$<U!;`<Vgx|B99`kQhAojvs9j?@+_5SsXR;NSt`#`d6vqvRGy{sER|=eJWJ(SD$i1RmddkKo~80Em1n6uOXXQA&r*4o%Cl6SrSdG5XQ@0(<yk7vQhAojvs9j?@+_5S{eQu;EWb+{qA4vcEo_?>|EKsjUPkL(&-@kqt1n#VUw!>wydSP-=_h{9H{L=?yKSG|e~ZQaHqY(y7K@I?>{zMCo;HW-mqq#Sd>?-gpQHcaId`Aw{YJa4=}i6Hhr!$y{q*s`Fy(yXE_z&)S#wWWl>g56@%Q*SeEp^O|6jZx?q~7t-Fq|jnC`yhxc>7xy_c{^k6B+hbkKk9YJYRYmri$H>*wFJ)}vnP<B)dsfm8Z9GXuLPAM58t3>q}3Yg-oOzw>?kJ$w#7x6JqOdwCtWF1;IHnZ9Mde826Vy<cM1JNL6t{$9s2pHuepI+l4~+0V=Ri|eERoHvGdTm2oo*HjG?f0ylPLqq-Fi_^|bU#^eS24)+&>Ys;ZP~ZOMl(H!Qo$urCDWCHfU*>C`eD$OLe!kD~)ch6P`s$bddi-_VpmqBDu6NhhU5^iEZBEeN|MIyFFrVDD-){Zfxz=-F3;lezc8}k$`uQ*e-?20FdD)<ueHDEkH)v&vF`|EqZ$GwR<plR6&EJ1%!<X(j2_Ci*%k-B~{@&kuy}i!->joE^{?+>`bAKgg;?Hl_qhIhg&EG-&@BQ8HD;qycORv=hlw5ZJ7TrDveDmZgFt>OfFtgIfz~}AO03WU10K8{%40v@?E#Ub|1A#|=UkKcN@-=Y7l}o_omLC9T4Y~##Z`B|8>75q9j~?s*HV@bZtiEmpFsyG7&|~l~K$8<oaoJg2W&-auI0`)dWioKPizRSzaTIVu$`)Ynd*gu3?ydyJRW}5Bnc{j&3=*2?vG~ai8{oNz2Z39<>VR`v_XQ3<;tOoOCKecHIvnWwbRU*6zsV8c&4kv#',
'J)07N3pR!WhsHhwwrmjqjMB~l+SK_?kA=_u_W(~X$9)y9-t{eTY~Ng9hg0o=v8TEL?LJwr$AV`^e*m61<^fzW2yw1p_|;-yvlnB5!TUc18q{{gcHf?A2i!6J6!5F&-vK|mjdggFP#O3A#;&@T9`jRu#sClg{1iB^(RN^;Wg);ChgSey_Tf77UQAmKJQlwVxbWBoVE-yrfVGAj16|il)8p$G`_h0%e`o+)u<Sap?*Q!M>l%@-fKCr~!%Uw}b_E{t2?Tz<H4E4?{!3ueetcf8U2+9I<~-W(1KgLG4xDlD3t-pUKLM*<!M^8M_O7nS?E4njzwBLm<^rb<M!wGOoN^Nw?Fl>0HnT?l&AR7w7q}xj2{^gWNnodg_`IyDReA$W_m#lC?)5wX+!@yrI5h<Jl=)#T_+e)31Sg>7)7yH?crX&{m$5HlB5<}p);ptT$|hjS)Y?E7<7_>qzx<&l@Z?h1Rr>Nh@W1q7R-V8nOALTP-Hd>mUhu0|w>IH>U+oNs{k)o<wG-GQAO7;HW-SX~g-eJJFJG_0=e)dp6n^(|bKv{HDc7a~yI(?_cv;gG*YVP8GyM8R!5pmniyNmN0(Uotf4`U;R2lebE9~=&rZ=!|FT%I=16tq5IzNBe`v&l*Mni#{r{eRT&#Z*~eLleL39wl|_~rAe6=A>6ovQll@!9J&u*+w+M!EwJ>}&^I*%a~W*~Dhfz+OLm0&M&R?DARFJy@q_ZmnQ{PfOb4^PfIC+XZ-j^;zJKr--Xhmn@3_PS}tAe%iNwJh0XK@W-dRZMfd2LDQE3?Qg-~(ux*V1g4&D3jC!H;!4`#aj=iHO_o@Xw8a*PKWSf%6+a)+`wFn@ImD5)=Bu^>Yh>ZR(<1k+1^T?fzNXplz;&h>JnpW?)SNkp->Fab;<{3AbvO*XF#IC$h}{I>t^mZ{)b&3g-lQ%+Q4Kh^Df}^YYFqe6>ezemv(#a~PXT_?4e>m+NB330&X3}NZ6Dw|Qk!>yzoj<(0C6ex{oAvEHEtlj',
'r6#u&{~g~9@hdgv0{k{L>MZs%HKH!;F*U5V`2Eo1$P1|<NAC%&S-U*Sf9L!7d;A=JF29H0%j@8E@j7|kybs<N@00h<b>O;iow#mXN3JW^nd{Dd;J$F5xNmYFxv$DTbKhwPv<un^?S^(lyP}=Z?r4X!OWLWlTiUT=*R*rmJ^g@wK|i72(2wX>($DC3iXYN1>8JEt`Z4{QelGo<aX^U+j1!C-j3bOIj59LsFb*kkiE)Z?i*bx`jd4!KJ;p&LE;3FsZZeKCuF5#exU0lr#%0E7#%&qL8P}CK&$!P#z`VdbA@c_Fh>};BXP9@Ghh$!2o>KA_^BD6Q^PJ3k%!5i^WS(T+WFD1ym3dalyUfGP%goa<Z!?c8d7XKld7nH$@&fV%g*T8#kXMjrNZvsnqVN*(6!I4G7|CnMa}?e~9z<S5o+Nn_d6dGd$g{}1$ipNrBTrL!8+jag9eJMQedK`(FC<SSZzPW-uarEKyi?(!<fY`P<gMhflGl>wD!i9Gn7o)gnY@`iTJmc0Y=w7|hm)6+r<1pn$4g#Mp0DtJ)&Z;wSSPS<U>(7_f^`P#4q1n=E>Y?f)-9}KSl6)5Vco+zh;<R`B-Tx=qgYq5&SKrgI*fH0>onGFtm9bMvCd=N_qGnys4w-UzSNicQeWyzeW@??rM}dc`uczP6|zs9qz1ndN`?UW_cYw0+tFAUh`-xma}T;5jng{luQU}piND)X=?$I}stFx|=Z|V)?%_G}n(E%-L9q=I#oxhb69!$4rfZJ)dmIC2;JKZeG1<_2X=eK4?+R&Fc;W97XtvxI&#64p4$sTfT$zXGkZK-J$KNH=yh*~}L(y7)1HF>A(r)O9wDF(g?~7@hJcXV}+rt3*DedT^&<$x9R2m1|>emH$<_MlEuYHh@>(v&Y!t>P)+$-Zb!3MFOcwVwW^W}I>ufZo9@jPFH>D8f6GuYSx`Y(es`OqyHq`9EZHZ<K{0~mBl{N0UuE%BUc!``j&_puDWI)~>W8*bVR9h%|KC3xPi',
'Vdg$Oht|mH4s>Bg@m+CWMx7es@AMdrn}_Ge8m&oy-pS}(0-jH5ls*T~=Qeguf$q#WNr%5*Wc<+#{Jk&ZDWQ0tvhh|&JTKY!dTr?2jEnc<xws|)!|=Rglg5Lgmopi<1<yG(SrLGBFgbMt_id7X1kWoqb^8s^12wG`1znYC-<{%lt@FF#dETanYvB3ArcdkOI?Wu1;Q6X%y3_c(X=c52;_se+{R^INY<6%;Auz22o`Y@fSO<T1%{-+(bXVrRyW#oA=JRG_AI%S6#Pd(hpL^jszZNc^;W^e8HGjkN@Gbf`gD%Zt(P=!#*y2QOe4a)6aXbgsvO?|qz&a=K_hu{y)xvXBEtl`d-y5_%8-{(i%vpfH3uIL>7yD<`upa*IhSjj?(BD{ndlb((x4Q5Uc50QE<)cUIO3(59Y3s%ppfj-kY(Df2)~jpb^Q?dR1<#4K&aZ>#KHF4U1YMa;qchMW+6=n}d$L)15PoQLZZvdzHaRZPU)Xw0#QNFRIfduv+77sb^|oDf6v7nSqr)I{uzgw}baD>$@%(kW#O|;wyRP-%e|A%|#dC(Ye~jmG+udA-=RDh$E);*CDzGKK*S^tZ*pK~SYdrteez7V1#s1JiJdfG_Q3T?HgLwyhjzf3~{LZ1pE&SaShv6Z3Ua`Z{2*e48Lut4UhX*a-*N(=YVBH-nC*b+?jtzFeza9JC#Pfa~XMT@;cHCAO>*jd5DgLgbV^&41vy;OD=&_ut?u1Uzsl_mSp3|rIu)j{TQ}O&{r>%40mrg(3hy6M||LLwCoy|MIE}a7>;yK68?=^ys)cM10h*!=dw?d!kys$q0UW4<FQLszrpXy<qoKwGt{kdq@;`3c9R9gv*>4fJQyEJw|Ty^=l8=hnBGP*YQ+hyJl;_pa*|0Dd-<ycc(uglHB(1E(V3WUG88ur59n{aiDhi=rh%6!BX*BWDCAFeHPupX{GauI)AKOZN4K5YS>GwixN7IDON%Ln*-0<MRxao?^N>)?6UuD_XJUtM1{#&x<D',
'd*J!aZkC@Qe!IEX#dWy_uEF0maEqRT=OepSPsj78-5TCT+;wYPAMwVmYZ9LK?bdG_{LyXLdiaOixJvLdx2fKr0p~13Ja=2X6#7uNZ!19O<hI@m*WtG18~B^su2qOjZa)U#`P*)XD<i(S9p5DW`}9`CFSqm2@LRW^t6@LgF71Uqx?R~Xe*bDR@`Bs76hDE74wOgv?|dJBkDtTO<@fM=c^$kiUMH`c_rd$(ee%A!4qO+m6W5LF$aUp9bKSWQ+!yW>_f762_f^?v?mO*(c0oI#-O!F`SF|(Q9qo{INjsHxOFLHVns!dRrytNS=qL0W`Vsv~`WgLB@k9D0{gi%7Kc-*P&!yip4k&Sfae{G!afES&aYn`+#vvsxF-|dVF^(~=G0w@j$2h3OMaD_SO~z5iRT*a)ca=EIxXd`sxGm#2<GK>(8TXk7m=~BQWZqyNQSu7&4D$~2kjzWWQ%c@q9%Ei(o|Acxc~HrV%#+NU%%d`|GS4b`mwA|ZnR!~~ZRT+$uQSgx?~?~eUO=9p@CNb-@(S_{$venH6kbA}Lf%3iBY6#Zj>3D$gUE}>lO%5<k5YIQc@}vWd6?v7<Y@|TBab7mBhQn(k33M}h2)9kjpUK!m6B(YcPc!Typ%kZyp=pw@>=p-h4+#NlNXaGlQ)w`OI}T$t?+L0aPo5Ubn<rcc**O@^A+CDI)HTn>jc&ftRq-gu+CuJA?py<B}$#bx`lNN>l)TMtb14ou`Xhr#JY)f6zeM1S**KQhp{eWoyNM2bsXzD)_JV^SO=;v^`*Ylm-<p)>Pvm8FZHFq)R+4DfARIV=Ec>Q`chx&OMR&?^`*Ylm-<p)>Pvn7?|P}vl~JE7qdr$geXfl9Tp9JbGU{_>)aS~m&y`W1E2BPFMt!b~`dk_HxiacYeW@??rM}dc`chx&OMR&?^`*Z4Uwf&~l>w^Hl~JE7qfwtLqdr$geXa~peXfl9Tp5k}Tp9JbGU{_>%JB&Gxiac=Wi<b5pDXh}nj<Xz_pcI(#S#l7',
'zLA(KF;n7miH{`Sm3T?w35h!;u9P@M;wKUtOROv}qe$X8iL)g(m1ra{bF0K=0*mBz6!jBW_*kO+dBH|`J6RINxo~N|(BGEkwG{aJeR;bT1!n&$Fl)TLefj<AJq5npAeZB$z^D5KrcM?3I7HxsZv_4(uk)@rPcFSB&Z$eUe=hKf{P{(DfpQs2Ps#5&W+m`ofWST30(T4$xJjH(m#*z4@Edvki{$ssk)Ja)U*P!L0za4QJ5a7y5Bcw%o(gO+Okmw{0+Zx*MPvx{ll$o6B+yK}q2$d`fzMV5ynRt_Z$p9m#C?{ma}_wRy})r-1oj&#u&wy~lA7-c3~3|KLF`NM8?o-ize%$=Sx4Xou@A-5j|luERbW%GZp9H|eTr>X3-fsqC-BUCfvbg1plI}00^5lDD+&{KQe@~Z%<RS_fg2|X9Azr7k)=Qn@$-UIu}=j%4hS42^Z^C6y9u;dFU<2|xWG9t1hx>mgg5rWKJqV$&&~fz=nwMW6Z?`^uvg&LhXT6_zs++LzxVo>&?USc5GK%lurT+deFgS37ib~wFXwx)KRNZg3U_!ct}A;?Z-LI@KC{+~xRI4uUAV~^vEG@jgxzG^6aJOat&wn_C#?ncfS#=M)kAUruRaiV`0}b)*Ozrg{CKg~PGEqr`{y&o{XNS~5!g=n^HZTaDt+oG?klbDkHQ^~bP#AS?(0cs5q}=9J1wHvOR<iR!bSXg*sr^A+cnS=mHvK5-2d-Z&;^y=PY{mzTUY37N`IXkD0Cp}gk9e~t_9w?jlcI>di#aA-dhDC{@gNZ1-xk{;@u51;cwRsg*{$<BmDE1=M_X)xDf-qd?;2#*_E~;OpFrO^K&cc&q{v^6n1ewOYHNxeb5<|o*mg4cqZ;c5l&u0FI9SSh0wp8s4v#_*emFkN{`MEaqCDR^gpGC_K9^qSO+@2()}0i0Do*Oet++2p|9DKR72=x)_)D$W&4o~PePZo?X0k$t)Aj~H}?|X_x)yJ4;!9Ck6F6Dl87s7TSfxE8zth>',
'sujs1o9u-yuyn-*aed2fiuGS|cd-a}w}c-q_!&C0(s>8^0KZ-(;{EJ#;`3&<5%xLVU#!>EXXk;F*F#rWI;k`CQl(#5wFQpb4xMJ{m?lEcGxDKW@6SIG`#3C1*!d8dhCcmO`0s#v;(q(B7jda~=?q}c#^U<Ae=XwkM?VYu_)xoDgx^r;sY=^76zkCDBj_+oTYe(!q}d>0*Nyvy0vmJ^{#Uma^m3)Oy`kqSt&!OYsM{xgKk+kfU|bdG;!2~hi+CM5^q$a7SyloD&lm9{&;`1~(n{0BeR<~z-BN|NLO12QN%+5Go`{!rai0OL`U$%+lb>U>Ycf!C`iaQ)m%vL(@-IL?Rg!ZEI>?fY)%d)U7h|xVC25U?ZtRgA{GsHx)7a0FJA=SON^Uq{Jxi{vh2E^>=dhQ+vl|e%OHR6*0FMYeD%t-F?5$*vH{w^xj!wuYC7Wkf1+L$Tyiv0H0`gADiktm~4(-Mrk*3aJeM@F-gFlsgHC24xq}H88`f@_NC>bI2cqK#oMgs>LAVHP%nSnT0^05W<b0r^+KwK(me|IymRTTI_Nt4gv=Oy)aKp$CB>o@pMNpi7RuQ(g%6HB7(5eG^_O~m{BUy1AXJ`LSniQ8OpUk=U0=UNJWS7Q8aj39k6(3uwJ??c=z&WZqEDt^8YpI7`S6Z+2LUv<zS7T*|wJXU;Z1J<?p>{;=9$8N*_iud0`UM}8s8P{37c@NgNc&%99;&0j@4-_x5g&w(h_K(Q_#Zz0u--^dS_!Kyz3+$(O&_x$u-*9m~-Nu4%6?Z&}eJ^gAgZN$C$Q3-axONrnM{#l#{JJ<M2zFmw#Rj^`V!tPF(BcYPpkpm|?1em6Y-x|UTWqiiytAkv4*Joe%$@kYqNhIC&!YRI@x4Vi&Lh4RT{K7DC^{97JY96K9j?D<S1;s;qVIc)_pRyxzb;ywEIxm>O(F2h%Q=E(OoHDO4G9KMF6whg=rTWg5Bpivb|3V@MNKLqjuh1%20ds|;z8&|iz0Ky?+1B9*IMKi',
'2VHBCV-2iZk$DpAzpykA`KK^f3*KM&>}T=c_opGA6kbmN&nY~A1O8KZw8tUf4^NO!3pcezd@cNTA9!-%Ldylf=}quGh2y_~T@?;nk998WdkA({*ySYRYGLa`i2H>N){O#I9}j;ojA{HmFxUdR)IzVl;7NrJ?U45hO`al-78Lfa2TcDJc3kkNHsWK!%_WGV1s5I)z3fph_-(=N`iKh!8+yQw3ziSVIuy(qj(AltsW0xg;PYnCBNz0qjQCd2B@_I-pw;GEz<Ql<9R*26i1!5%v%yCT{9F*13S4F)&lQ-z!S}o=Zi_hgCSyJ9^3B5+mw`9@#DAY_g70~AurKthZ?*|O_-6Gu@SZmdKEoWto2lJ#eQ!p+hy8goz(wecKe`8hd(&zW?E6jKdO<>W{9Jrbl`p{i-c+<3E$HJM@Z>i}rqJ)^zaF#%n05vEJO54)_;&tJpJLtf4{rx=$lv}1`<?%tJ@Q5Vf+*Zy{*-!%WBDUm;rsIYwZ;17cW#FGncu7i;zNF|K<K*jV+^p*`2p9D16{ve1T^mk{+(CoWi52qKfy2ZejSE-Chw9f>Ylu#TM<|CcGiI1<$ZS+>yo#i0sJLz@(J+gywBrcZ+U%IV}0^EnV}xdYupoll$UZC@i{Nj9=hT@Us31eIjn+x=NUdg{qZ{23-5dVqy^&a>l+i`zpu}17C+y26~F&_<7@ck>t&AM6R&4f!Tr4+Q_EA(?Uq8H-sL0kvDeM}fv>-=H3WX}I{Gv5dHy4WUfp>Fc=2nK;i%Vg^9BhWd|E&7klb6{Veh%;I>29Y_cw!nI(Jj;LBQqlu+!X`p+Yx5)*HMkcc4AiA@?I===5`&zd_u~t@RRhLvFP2n_U0f-~+kNSK!yVCKvEMIeBNWJ~?Tp5npm{orYdM=iGVNLC*fmu!EdUcfiAQmZ!mQa%R4P9p;R)K|dkq)5;yh96%l9ft;4Tp?}Y*JsWW_C-xBHM^0cS;#-beF!m?MycaOLU<>vo`*{KEC;MJw#NF(l*CQWi',
'AF)6@%-%i(ev-ZF3G6F--ba{o$)5BG>ykZupqN|eZGt?U-C+&#Z+4?5i1*nk2C%#Ah{N!&Y~Lxk-fYK?;7QrWiMao)ya2@atfv*gJF@P0W1X^o3P)bcI@|#9CTsg(=+CoOZ-ifG%}<B@WKF4w{mmM&8vB^l-vRj~>%%#yo3mO}6muH24<cS>#q|Z>%L)lb-p%sNgP&yCUdQ@n8Jvb6X6Bwnp2<wR2fN6;Ybfwy6233<*ktg_%-xTnXV2Ww7W_VQ*<JAW%-O@tfD`@1Jjw7U$bXrA58e=SB^xo%klAuG=6o{io<tubGbsmoD>E_yeTYo|`LLr*cXQN5nbvdR-<by0ux}Z!Gamq-pM?Kp+~1D8oN;Xr>bQ(^*Aag*j@Tj|W$f+>JI&a55%pWf%C^Yc8S@Ll7c!>pgPmlIpNBbvjNxA)?q&2}K1r0J7mznHI=CZ0WHkE%^G+FcEwEl0x~<~-V+LcrGD2IUACOV04eTw$eHeIqhTRY1y3AZqH>a1*M_(#EFBX25{_6E2QSRP^eWm|;2l*!bh6ekUez7iiLi*`#$am>Sl95l+_dY}3N#A||*OR_+yZF4-=MIQDAye_+3won(mOe|^QTo&+u#5CB`XGL!kM08dNFP2D{-6HoK{1EgFF;_Aomi*zk9s2>rFW_WJ56uf0P!xp#V~O{jnBc()9cqoJ)2(p1?sT$8iKc_CvQg_NsqsbK0<nI1@QCqs99M5^zbn7;`ET*D`NgB4R)FyU;_S;Ua1-UJ-y-y@pG?_#phOt0<TMV4;FJ%u1!$yr8|EEzfO1ZLcUFRI0Ju8xBnJ#E8T9L_&nR&;{I#}-%7W+&`sc1Vqa{Ae<bkhe1Z4G_t>^=D?Uf_rNCR_KJ0%--Iea>(Mx>)7l=>kE}s74`X28V*ZW%@(9=@<+`A|ETDngT;(5B?PV`UG14Vx+J@_Q%xzfYjk$=-8zo`$5X)EqGzUE-D&YfMvx^G4uoBn=;*#G(uZ;5%b^WyhgJ^}AbZyy7`l-_wa',
'>W}p9y^)vF`!vTKO8UV5<Hh{hfrr9gYB&n}$peo_|LOte9n-(of*+<YZY9=j<z2Dg>lcd8**00sIq%&hzW=B(_(uBKxyb+NS6YkpxzkX5&!gdu#C+QgVJCV0us#{uSoG&IEMrjjWjOZ~{#fxM;&(<!KkQFNYzq1g88vFdjxrjKL;T8UlY#u2(RIG~_fH0hxG-W2;&;ZE$H1>L<|e{kGQLR_e!l6fn5Wx!A3P=FbSV5O<LbuCVy>>U_`Hm|;58W~-Qo9{R=dT#U4<mrOJ;B`>eS5m2dEP>>lq<$X148s@6GJ-5dHJaVc#MTWq$cJ@_*)nA27d{xy}anmAPlJh?{5Hg3n~$s)v3^=Cl6bJ(-0kz<aW+8{_9$70totvLf^G^Q`yWz#p?(_XXd{>Y09A<d-$*2V~8dfq0y?axMHPYi9=5C+l?IOp(7_5s$N8<-nh_4Yi2-*{&%fPlkSvc$HnF9@ag(wG-|)yN?~}x$Ln?;(hbxqu$Hj=q28N_$una>}&hp6Z!iB;!}=+)dG>{N5Fn^BAl_lIq%<tpX7A9j5<GOa2EQ>In$~m@8qo7EBM2Yop2pFS3<F`InP8pE7za_;z_Q@;tauW9Ffm+8}5eP=5`+s`^X(V9QK~OU_SU~?$$fF&fK&05&v=@{Dyhn+@j_1qt`Cu1rLmzj(YxegG1;SzW&%*JU3zNT=?DVC9%jKuXh!pANu-Y8uG~NXZheqc_vlF^BH`<KwipAE=K&wYqtb(CvQl5^qKNzCxV~nZA!-d<eljPewX*?JH+99tr_BXzSnfTKR+=N{+Zt<3w3J#;9tOl^JiZ{-!gx5I=(0Wyx>*&Y2(1l-WcZ$0s4J`^?p+`9R1HXopa4Z{c#Py|7OW0tjC+Zj|Yo-#!>uUZU^)~3mo?$t`t;_TqWii4<n8g4Cs#iDVXUa>aa~&@T-FJx5fLO-ROZj&phq4r~_>f2Mg=f#r_udoGa!Szs$n-7q0CFzEybg$pvxhGXnlxWLyz(w#fhLdQm4Y',
'1}`k?-e1(w6Pv=$i@vLY-zz#%y^}aaYAEIxjk+Pu75h&S{fF9n!H0^w=O8{7PfA2wE?)E1WO3Z^Xo#3^tc&|DG1&<NEC~t}{hqq3ke5q(`HP=V*(~}-8<J~?KGmgV;+UkjIA60g6@C2B4WcjKsEIf~==U1DwsgiyaUQUxu{hWJxj^(^)7KxwVV_-FoTqA{jK#TT^PS=laL|Y1{Bo{2&W|-aw~qu~ZG-bzO-_D296LHMumx6&fF4BK=J*ld@D@0K)-Jp=7P$9=Yrvbga9*r^)98*k$K8vc8zfek4(u@WbKr=RI4?F>YKik~gZ*_s0p1yY3|P3V7mhzG?7}&*VbYQBfE^Fx+}&{GHe8qC(ywtthWmTqdJOM`;+)&C;6Bb*jXWkne_)j8j^8tCzswQ%xixe*MvMB1^X@%+aZYP=J#!E6wQGVnKd-z57!v^fk8w+DE8wTULC0V`duft5hpz#>kMU0z#rs~=!F`)ptl15$`~vrHQrBY!@Z$)4zsVONINvq-)*>Hx=rVK(CU-|U0}Cwid#0{qO@T4jg?^xg9drz)1H$n8rqiqR1a7b%3_N);9r$1{&XY|`@^DUWR-q%#!_DHp!8yEH%j3`snGHAz`!Jif7VkG(_c3%7X2%WiJ!Zd-f_<31`31TGb7xcNHq0Yk`vMyp;=JGd<8w2C<N878V!q^=(1-4>2|bhf#lf%-^QU8=hp;g0`V-L02kT)`?FZ<YESmXm1orNe4V*9q=kpdz$Cv?kHNg3}#f29+z{fr8fW^CUKbFq7wZO2e*dNRHS7U!IKd8G4IOGWQ6_(SjvHzB<!?E6$KURXS#q!cK=(H?TCrtttXZL|l#VHzkB&(1*@Gq+xp}0<~)(@bMu<G0AXW+OK&^uVoPlw-FZG0X8JiG_jVRf~Ycz@ba@xG!GtgE%XCH&9YKg}B$KM!_p-N+WYD(es1L$_c(Xb`T)dQu1IFsv8Z>3|#O!5^#-q(WC@ebE%UF6;ZRp~JAw+5!KtF-j78ALp;}',
'bDO|DSU;P%_2TFCdMpCA%Y^R1rgwelhHO6X550=b)Xva>*)0A)?7dZ#R9DwETwc`z-2}Jb9^74mJ0ZBc1`QH|1}DMYA-G#`2oNB_g1cJ?(%oovb(i)3K69Q~V?6H|-_7^md>7Qs8r@x8b<RF(FPn4il?m6EeB(Os7n1MyHG}*sGwwI}&2aEIl0QGS3G!oI@?p(OvA$7J2ZoW4v>N*=sz9^CkY%o4B3~&Bcs@~0OOeOaF&*}ERNvEhu2DZ!#P>!`9EbIZnmr5p5w*NGKW|eM_%l(zjlsG{9Xrh5pF4s3in=iue2}QeMcP5WUNivm(|PcoqJrnr!;HG-9wN^w?*hn-Q=muDIe!B`EV|%k@Y147_4Ptl@{WhB)fD;@-Do)Zy)Ak|Z=*Zp!2XQxz6EwJ`n%v($RU|wN1{i%pzqN?9g8LJ>uY|`%sH?t(et<SbCxc^bB|tK7yBrB{dw%W=&k9mlQ&j^-@h+Y9>_zN!EcK`_6_<8qED`V4f)sauuIYBH^MGQU+M@u7=85x|Nmw=tbg>Kw&24=->;8-68$Iwb~gISBCgM~Pv9LzzsMQ}`7#6SNA#;_T%R{1`S+NApkL8(UuK7Vn|n8966{X&yN%G7==a&lgM0s-i*m|J${}30_r;g<{U--d=1R-YX?cO4-yVFl=-ATb`Tb9^f1+P^{E64Mi}(BDCco$TMeuZ@pDqDkEc$VI*pui7JNS8b6D{QBrG*|v-w5a5uUv(HioQ62_i^qy_2jQY^SRCq`Mi(S;O8Drd5X{Fz+aHNOJZL{Z<`AH7rk*W_+-&*4(^0pu>kfvdU1W;-`u<8iO;A;Ib}k7>dVT1sXz0<zl<K*4SqR#Ks5GKbgu<`{+$ylK(;LjUSD*xhWxqV*IdV1slkhiu6z=9F1k!3*sJIw2WCR%O#;6uI$KWaL;5_hW6>$X_*`7)_<5$@xSuG`D-Yz`I^6d!rh?ZO^<Wd;A9Z~*{7BUKU*JEY{%Fj7e!$KBxML)Ercvu&fPWRWthfVm',
'PG|U~s3}8gK#uN%^^f|#Ce|scw+-<`RELeU3(bnauSeBg4xU_8<#)eAmdwZN%3mA%ASz2ket)WR(;&l=!yYFOoJHJ}{6k+m<cs(u$a}qEkCR{elXmk|1md6M2g*%`-1^NM@++Ic?<AjB4Sdw(Q_{okC;#z2d6@mDA`VF2B?sc6<jpr?-z2Y_0ro3-#o>sDk{3AzznVOUAM2GoZFb7=QdqBKfimDrCVQ6;dYbH+3(qUrjhoPqWM}3<FOnUuHUV<mWAZy!^#gA)*__xS^e=6}w@WtccYI&6-qfRHZJU8Novh(6;K?PcvKM-qtk{)T^j}YELZ*3+cq^IfF7|Dt_cZi8GG;CK+mR24!Hz~=t^)fVc_ILR9=T^1_G{$2)=eQ7B+^e#nMwZWh>RoXZ&$z%MYc;1`yJWvXYA|9%5KE9kwse|E{x2+y8!*S9r_k&Z?Xb@J$_+Y`gz7x5qFZqK1H0Xiv1pOxIfk}V(Wrf@=*^mewcA<J>;18h|3}dIx<0aP76C5(KILGqloH-;LjpTmH^K>B6k_we?;0c;FCtUN@JbE6N|%dhQBTdJqo{<>ont=OwiNtqhBB%58v*99SL9I1%EbtW*qie_}C}d|KS6#lmFWJFT{7@O%Fpa!mIDVa||!Jj^C4e>1D{Yb75!0UDFw-B~F1o4|_ck_9N`xMDk)UOop8cJNh&Hf7tf9ut#AlSHQo9&Du&{?YN`hCx;EX2EP>6B?kK^tXUZN%V9MOK!3tYf5XqqI|^}NSo&?ScVXdA@myT~%#2H8+tp<JxdzY2btw-2?>bS1@7uiy@s4Ya2kYRP-xhJOYtorcki*L(o^kcs3%<UqO)11ZuKLFi_qZxGhCg)`ehIyEWt-ZIeBNT%-!9uz#GlUiWza9@%l6P8=iPj`Kj--Xc(=}@cMzvIcN~NNcdp)nxW+kmBle+l;uh$EbJ%|9fwR{|=&`d+9Qel0hUsB{oK+ga9yyE6z<zY*ygiZee?i3O&amm-kr((95YIW{`oZow',
'9=}Dr;kY(36Zyoc@O_R0`w>Stw)7}XesOlJk7L#=@{h+KfgU@CF2%YydQHZ@cC;OX_|eg5JmM=ywR!N5j*`3Ln9tk?zuS>1CBG-C1%I||1P|XH?_hrPY9Rcr{r)TPwC$Js!=KnsS#6N}*FaD0TfV`*v9C-Hzhj?s3*Td(v;*sBA2AQtXYV%!{>I*UD(sED`C@)g-F@H>+bcbVf3O$lzOm=&20pGm<6-E#JvtlwhTS&r0P?)VG>DgMu^Vt5wx_kKGY@pb-`URZ#(uFKAC2d1+uIWM(zdxO_JM6>IoNUAysFSQ+te25E3l0o3HxCiybFAFThC9}*S7XGp?9{X8}Yntb<#t>ZIu>RgDjDEC}h4ralf`K-{bk%QWr&hXbTS_{;-+u;Ad<6DXfDv<`}-m`t%IqKI`o#urJn2;Zq<_R~yGXc`EFib=PCWYu1f5u`jJFjxdj&UzPmspKoHnSbrM9|No&7?4Y&(d+ZBqcg9=RcI&Wzt=}$yeYe(I2p+k$+6Kf4*0O)W?plk)!=G65mf-uc{DeH!n&uJwpEY@X*n6w}4D5|1&=j9<`4k7gZHZY3yJvac1b*3aKRwpja_v3%?UwVmq2HF17u&F&a1A_k%bu6eOUu>>#C4XnmGHfmWn=MumU&mOZkFjq5C>Z(u4MgTbY@(?W$0$`;4S?bw^@3;Lq2EexCws1(rO^`J<B(>kq27p6#~E8QawBC9u;$8zgbF^g5R|iZVvlk$vbg2WVSz`r<RP#p~sd~J$XIJE;omC*5v=0&Lhq<d)q^=%%4oqV{_~d=$-lBzPJwalQO9HnD3>)^E6*i#OIkW#x*8?{yprh`J@HcX+D}2{C)HO27KSHN$(-IT;%7j&jJ5yUNsSYE#@U|_+j(>L9k=ynE}>&rp-hCVV=mo0`r(Ru+QcZn-LeAhYZE@Fb`<Nx>28!Bgog!i#*ZXDG&CoxovS?cZ>S4pXP74zUBsNu)gNnPZA-kf5p$MJRbUQF89LAI#qM*A9K;Gh&#*$',
'Tfr`v^Ty)2n{!S_o^Q@t2JyK$<6GDzbK2ePH%K`Sd7n9|E%v)PqAL8E*;$<HWi4Wm=*z&)nSAw7&oX&>qmRP$aW3@7^!5z)jVZ>({(+ZG@p-1_8`(eb#6>>;gYQwNFx`2DejU?I<8!WldI))G8sa3=g?xwyOlPjaUzko$ehT?VP3XPpSaPhh>F^WO155{wVtq|}*Px%qw0kb%J=2cquqURivype1Hm%_Qum279(X{3|;!D%2U?cWF6h$A6Y3X<DYgn`!al2`OX)x=DEnr7Xv-ZccPa<0?$e-u(zNRL_dYUH9hP^jU$b`NO)3{yia~RVQ{cNU@9}$O`My$quGYxM6yJ{Mmx+LV_d+d`KxC!xuss9)}chh(6*cadDYdlX=ucExZ9=Ty(P2F-}|CzevK_8H*Q*q>brVh3E{MvOw9%gDoJu<c0%5`Y*Aewz0sSsD0zWoOJX8L9!?7pe-Gv05bQrJJH2D8{tQt#bL_JOoO{BEj!3iVP`tum~?*4WKHlIq2w_oiwG5#O7t)}-E4zQgOQGzfh|riv-}edYJTE}P1A;{TWZ0`Zoq^cn0YQ>h7v8%!l@mSf*ZGOky#J6!LgTX?-i#={Pn3U|OhHx;T1zh^3#7k0(;RjPm3ui}8+HRbbDPx5+@pP2G^_#AWlwo=-uAGuPZK4Hq43;C=mM};KzyR=08)0AyE*FWoW_y<##KiSvv<wsu6mpNfCOqpAw{%OiIcNzO+u5iO;{DS>68CoKLGo@dRzClyExWkl{sn==e!JnGa#7w7Tf0-%u2KM))4(B?j8p!KSd6(;&vKIQQOkW&mKv@8G*OcPdU6h%Yv!1?q6=izrL)5b0D6><~lW(DZCNGV8s43ZL?!RO$xc?(xCsK}ENST596S0r?AMx#5_5rzhU*U^9l$BVQ4|~D;3Y&F}vLc_eD<%`=a@r49W8R<3j=aI-JTjJY2-neBgnHqK<NrCf5f5<m<NkFN;dR;FEh!K4{_TthP4))WDbvA^nQZ?MKd|lL',
'=i0{eK5fmo4{dq*cdMW4YQ4()vu=u^{Anm9>q{nU={}Sx(SK{Qyl+Q&nfhqinT&FFddk6EZ%gxGl;vobELmxXERGuNqlz6vd1Eu>@5D*W8wre<=T4*?Q;4$92FjMNC~NZhm`hTx&AEzFrtVDXs7wdpp}jV};{BQK8^FtD+7;6&UYF?rpNnZ5^~S_J$7EW}{brg;yJVU~Jv5Crem|7^)6|cBWTqap&!&!CA5*JElug-(W@^CaWU9?|GgYJBn<{>yENgsDN#nYT@IFlisUN0%lPPnvU(S@n_`a;h-!pfj%s{(kO2_A5O2c(Gr7BPPh4DSniz$=)DU<QJ{7<G%8UBPajCx{naetYd#$0vB9YfOo{7)u8X*1T#mXXpr&ydD-T8zJ2noydJ?>8IkYaU?8Qiik{ZF9E`X*@^M977H_WJg0bFk}To8UceTry-5&Fr_wRlp!Mw=`^InkXA!l3~By9$p4=&O9tb4(4Lv{JTz{H_Qo`Zafj)5#s%gu?o)F|+FkQ0{%k2x-dJYt2WuJbL+ju4<F@bUXY3gUQeLB-ax6T_yYESR<Enn0kG~+}fbeYm-iT~m*T}CJ|0Ju<?@QjJ3|DhL?PBz$0M#w^4%+wbyq{DjM^P;caG$4L@tXc5GyQ0W4V0NG@co$|@O!gN;r`E7nSM5hp(i;n^1RmE>$A{LPT_vfKa_o;U-f6dXQ4sV-y&oAeZ>~Pq@O-Ye^M%j&$(=_Nc#Jp^z#+>@j9wR(f(H(#CWCpW5zYLS~AY4bC>sBzX$C|!yx0&CMy|Fd|Qv})51i%-s(8&LZ-IU=#M+J=el;Tkim$X7$5XZO*_{o1?_LYG>i`i=B8eMUzz@8SZD6H5i@u{qfY(JIL}5qG@$|GkIAdJucrkV=gsKC=R5l<?f?9G+$W3vraxNNlKAw>SnA1|X^7KJ8;T;{H*J2(|KGlJ2;<y#)SG?zXon95aE^fK=snuY6YPgKo!*IhiRtV{oDX37dlU8b%5LPtrW>c5F)n}1`0t^O',
'`}8SsBGZd*z$;C!f2DrKy`+EqSenn#GoA6aFE)ak^xGx0CZ~Sn_NlOv{m4hLugzc7Ks;bhbCda2#{NZ^r$lFG-f}Pt+kktc4s9-!1Af|E>>2Z+(pwSNm@AAzeaKw3*<HG;Qp|VieF-~hZk!DE)7;EXT)(xA-`^n|``p|$9rV`Rs}S_T+^-($yXL|DflHZ(FJ*o->LT>mJU$Hhk9kTx*gf;~c~~#=oafjl=0z3wy~~&I{@0kw(%trD-nH#9@@(^-Hbu!JxKDn;@7;L+CqFSyJUb2bZ}X)*eVNxXY&G8<f_TsTI5+B0=6@c*ewp7a#T(4;J7S-hJq5^9@CWcbESBrA`xe(8+@B@+!Z_yNKi#4={N4y3`XE2C<nD=mX!)uy{I#XnP}nm|nTgmZmP$*vGw<Jz`o5*^?N5f6LA}t@q9FW>rCmqdpQX!U@*#TN#(uX9$W)Z|hHgC1YUEz{P0RRj=)GlXSJa~{Gf(k-3-Yl(u#9<~WsMd4&a!DZ;&RK54~S<i`v&28SdPTMVSQvQ^xbkUsszKSUwOZ`$|K&gJh=Ugb(SCb_cz&_F#Ng*zi&wx%{od_aS!V?vCs=^<kmtA*ZM;*tr<$ezFV_dO7UEPt5`>Cq4mh;ttH1GAF@_xi-op+T@m_ct(zTsY;6*W{c3HIi2T6X{x$5Cwc8`uBWvH=@JH6cH(<xDBW@z!w~o7ueQTZal=Y;UvA{8{3;ah}=Zb<owyw*?y42Q6h*PY4+VHv$jlgrWo>&3@Ydv#{@4pm_=ViT>9_wR$SO<Bg^`9U6vi`OcdSU(e5_ZJu%MLqYvvlP73t_8@K}J7C+-XafXC>=<{ZUV|<vsu$!&b<Ke9%_1Dc0LoVGZny?dy2>S6jWh&|BL#t9gDyD=+ygoxa`8I^$00k8MD5>`&XUe!s9Dc?o{NHmMlmINS6Ehgi4t<N4Z_b-{Ys)|^Lu$hM^f>!Z6@qAq1S6b`>=J24FTneEJT)WK|*zd=37cIyPz!S<*K?6U2}YS=4V',
'Tng4z-9I7jxA{MHX8pB0;#hmcZPbtKDeG2dSa1Y-XwO~<@v1%FD&&FoqF=xc*vm}7_uDJGVaM&Yy0iY<ko8J?^O{&^d%L};&)d7@!2R0$&Vs($2M1uU>_7HIo?`#$S}VwD6>uN+*=tePurCfr{9s=-0D5KLaE;foqa^C)_I>lQ&+JD(v5tMZG5_zvX4nb)bu0F%{a!1qul?yZ*0)~;$tQZxc-)?_82V;UdWQOk!<m0EquPFWzK+y8j<U`k3%lXSnV08;6zGV$kfYc<oO|IYdm4VnQRM^H-BCN^&#d29MLpHgyfe=kX*ZVj{H{x|za4#c@bd;8hrM?E@Hh0`G4=-Q|C4Vc-gV5lh4psKzk+?@SauqI*0F}TievLi*mK9O2|nTuU9nz{W0kfMk4OoB<GAo-Epdv?u)B`CeTh3f$p^h~{QGDw)6$vnvkrF|UYGA4>Y7f=ux!LZA`z!Jqn1D)oN2P6uI<dUinvRTbf~L3^G${Qa2EamyWlL<6m=qJg{|*-9!rooOPyM<ug*qO`2J?c|0F*14%hGOoR**8vlQ0F*{>d+hx7ZEu+PpBZ4eJQ$2LP=<eXHaDS251upZ7iF2qgFh4+xRIG3-*es`|vGMT)sw8WXVpWyfW)~OP+9k+2#-x%np^HdOa$9Zlb?1J;s9oTQ@jZ#yHOHIb}bw0WPd++?m!RvTk9RA7q_FFtR=ciuS&(6ewapZ@6hjnpT+XCltxhi2_xRRxW{<u=yMcm*@Gw&&RWK|JIyRtln9dP9wh&a@h_Y>kcSHWKRURTje*blB!dGI~1as%5FciW16<EnOxpIh@2^vhMxjQesmOd`JbO)U1ItN8`^H&^SW#P!;f_vGsA;&pdhe~36=3F3WycOfo!4af{Vat$61KjRwu2l6!64_?Ggu2I<$AGyYshMjbcFVAy*CNaNoO^qBtUfdnzjjkE<c;B-tz>d4-UWHz}7JLi4;97JN&)2my%WIZndctqHRxamtuKxW2@yZLRySmn$CeLoe',
'X4FYsn||bdZ7GdB(Y5s<&tu#^0Qr$?M;z+(u3gP@693$Yc;2<=AJ~1@-ei1k`@ZD5@6XVJG6{O&I&hIZz=M<EXIzK!lJ|FL58?^e;XGL>#~-CUYo?6HesCR$ByM^nf_icI{V>X7d_ISVjHLX6>vw437E0?jesA+*ypCCid0o4R%kDdHk=MT-_Q>_yh;_WL61RB2H>fW=+Yp!C{wJTuwzT}7Ee!{A{d$+<diLRSUDue;V@*ct>8f*xGh8d$7v*!h3IE|*T#V;9Ef~c0nY)0`Yt{<X4_(uz0^fB_Yl(QrH8~=j`m*W=>Q6fO6W7S^sc*w~W52q-zt)C&c$3%L_XwY7&k3+cuC9gnbB9Cxo;G=#P~V54{^Dw~DL3)z!(4}2J9xj<Ci3?c%TFes^D6ASt5`kc2d=NyqAupj{Q&WgE2{_jfGfiX<Oi-)f5Shzl2575{hovS(!2uu-|0>4L_1Le>*$Pah<MKVqIPTAmu%Q?&Rch}-<_8SQJ#54yK<s3@$iE~Xdib?r9Im;k~sM42GBp}q9Ech=d9_po0Gk$qdUjcM19;jbO7QK=Xb*q|2w;ONJYDx1985y$)C7xXYE?ZW1N-N@qHzpfJfuZ?<dZl^%L(S%~{&b$dNDUPm<BUc}7CN95H`gq(36B!*SDte8_SB9`ZBC@nyJf$6lTv<=C{p5OMWr+=pXMP1qmDq*l;-$B4RFSyoNU@9Fdh&-H3nH$DCIcKXfo@31c%g`@a=*&`4iI8r^I#JFGy@>RRH;571!mm^NFKYWePwO>hv>$0EB1p8s%Yl6Pp*PkH1zNjPPkg4~4jDJd8VO-Sv4&$ZKnHWd)<GEs$D<Ge?7ycJ<sy*v~0>sa6GHx@chdr^qFNbxpJt+%&V7ro<IQa2P^;wPU?jufqANiMUW?|ymqdPTbIf!=6*0z0f#<%&&*Dia#B+JHaVK;1PPV(F`dl=6n`&bZu*!rYMWya@UkPm+3D(=U+c@X@Hb>Vy5zjb0`<|l(@z@Aw<Y+;;U',
'e<SmhvJ;tS<gSMAw|?=+#yqDj@#Z%N7co!r5Enj`rVz_qsh5(+?q)tYdH;_rN4G{iU}^twIP<~k#HmaE{G55^UgFRZCt%0T@jG%5za7K;`$921AM?I5*hl8o)tK-8yn^Q`4|!OH<u^0p2y@*q=Fi1HLBGwJjXHyKIO{O)Q?TB9cP8sECqJ_OxUKwt;(5J@*NhpRpLL(%tZO!DMVzW!*8aqEF0yVL(d8-6{dvrTK<-!PIVq=R^W31Vr@QmKm8a<YNgDYt&jaatljjZ9-^O#pO7uq`O;V-|JnzT4gXd|yDV&Grcq|^y<DufvZxUQxxI6nM+Hme{|H0^w2sR&%ex6{(o;*(^XUzpX-y#Lik9dEU=S1A@kG`D1G22({+n9ttv%t>@JO^S>EA-(9T5dw$SD?}>oR1R7m7eEgM3g<tb0F%K;kgeD?(&?6YLPr9Dlht}{Zma?7k|I==<o4=J9P=qXDE-pUVoOy`PrZHLx0ZWa`K#n%TsUjd<8SlJ6PHu{cygqSJ9{B>;Bb6$a<sD-{UKC8ht&!H1;of9zq55uX!K$Mc=#k^n9Lwuz5fGab{jcU%B`Dm*|`Jwo2gND_hZ*;>{UWp63>X<D8bnI2YC}@tQRq<bE&a_9ZThL!V#bxF_h3O6-0G=RG9WJAwO3EV2X7Eiv_S^d}~Ge`f#xqaV-@lW?jh`ezb0wZwTc2{Y>Q_k$~Oo?6QiI4>ljVj)~-LiPeYm%#be7@iAI^a<pZ3g|cX?5l@9YR}S+=x_6k9f>|ePq!7&BTwDG(C6wY?16rGQWwFx#d`<he3SS`hoPVGr=!@Xzqv2^i{ocr;~cpmHPL?>-}>Yg&X;QdJ&Dit6n)O|5fjj-=6+usecJAu?{GeX`^X{mQ@MYghwE`q8Hvwv_Ztj-c7OXLp1Zpo`%c|i5C6<QdpG+2KE+nWIdh+`t+>hlwkY%$ep<z2l|D_*kI(z`-3{!APv6c*KkTRSz0rsEDf_o4*)QJ=*Zc8ZpEHoR7UQ{p{QV*R',
'|Ks{{IB)3V^c~pOABR-LISL=!KF9O^SbZ(~-oF}yzTl6kzGt7ie**T`hvz%kZ+^jpex?t5+u(jbEWgKo#7RG(U-Lu%lI&A&nH=XUeW>bW|6=|xId3p!GxQa|_iw@Z9q(V{NWngJuH*ZIJSXA(nuh52dOyA19rmO5LI2<TPTO#9%=`K&VMpGVTF(CSZ1qg+)6C4iZ%^t>?B^^7`}FSbA?*7;@DS_wZf(EGoU51%dhu?g4d?v5>luN4{_fjyVeC7djXtz@1;1Fze()pcXMg9I&Hn9=Gue;(_#pd#FD65u`P;+O@qKSMmBIhNT^P;&+li^r@BemiP4<y@T8nd6-Zn0R`FC$Cy}rW!;A;)oSNxiNwvk0}o<N*$eG|wx^>98_+`VkL-njF*(I*ynxE-En+?G?YdvVL!!LGzj&x^jgxY4=Lw;I>KDefb#<6f*o+&6VNzp{D?oQn`wDiZy5ae2!_U*a;X;NO##!oH6)SzzyCKU(qr*cWACFJkYkL%(F~-?d<WV^4gE{>|9EU$x^YiUVLTW0$|deaFsOj{A$9JZS^u$PJvoIWP(Rld(N!a&Bk4mUzywO<UqVW9!U;Ud2|i!9K*6+JSz~*n)H6e`0g)<mY8_^7>LPLVsOsL>HX16l?8)eG=nakN(1#Pg!AKV`8r3?=k=UfpZ;V9>(B$Vs1A^KW5C8ci6u%=g&jmV@^LoUwh2);#kj^Lnqd7{^-x>YmeDE3+o=U<qGUU%!Wodw<~6i3w|nQMUeAMmllPd#w=Wk^JZe^)x&d+nOz)qEoMdwoI?@w^Fj23#!PL2e$tr9#n49_GqEnt35glM0_PpX{8R+~C1z~WkCYMooH5_xe7%^_cVa1*-TQB8{9f<V|LJq|x%wV`udYMarR&sn>ppZ}x=-D=)<Nr{b<(<N9ks4nXRW)QgPx0?lb)NNqn@jtv!1)52hxj>o=9(`N75_lne<M2D7}=PN^hmd(rf9t^j`Zw`$GFf`$qdn`%3#v`_9;h+Lxhys(q_{',
'tbMJ0u6-{%AiE$tA-f?vBD*3xBfBFzB)eqTDcP-%9g|&?os->@9h6;^os`{_9hF^`ot51+?6B-|$WF^{%Z|&g%g)R0%MZve$WO>`$d4F)MSdpacjSlUm*l79x8%p<*W~Bq_v8l+zbHQ$@|*Ib@~iT*^1JfG^2_qm^4o?VmtPP0dHH?C0g4M0Cn#=E9HF?vh%*#-gyImzC5lrNw<wNLT%$NgagPxPDJ}}dNs5~kM=7pSoTa!+ahT#VBTiG?7K-B(*D20Z+^0BDaiQWw#f?TBskkx}XDaSg9ICifajN20#j!?Qt2j3l_bLunT&y@*akJuR#np<ljksHJcqlGcoUXWCalGPs#rcZ+jXXekK`2j9-k>}}d4=)}<sHgHl$RKJit?6F9;3WQd5-cP<w44elqVT^lk%ugUZp%sd6)7q<z>p#l(#95Gx9p+d7->dd7$z_<%!B0l}9SCRGw+%oytQ)d8zVL<*mwNmDeiIRo<&S*vN~OCx`N8<<ZKkm1isORvxarTzR_k_W#M_mDm4Qp0B)LbpX``R3}i~Ky?Jw6;x+1>JF+ygz6HiQ>bpCI)>^Rs&lCBp*o1_BC3-ZbraQ5LUk3@SyXpX9Y%E-)oE0>Q5{Eh9o2bM_c7`~stbkcM5-I9j-<Mh>P)ITsSc&Ol<HKfTd9s^)U{OS3e~+-2UA^4bu!h>R7X=?O?5Wa-BgEDT~2j6)$NQrp6Ys`I-lx(sspMns5+tQhN>f~uBbYr>W->IsxGNIrRtWdW2&xc)Hzl64AntZ7ge29byL+*RaaG=RdrX@VO5t^omO>Q)p1qVRh?ILU)6zC7gn9vs2i(}9I7j;&aAq#>d>l7t4^)Dwd&ZaYpc$!y0_}!s*9^muDZGE=&Gx$&aS$<>hP+|t4^=Fz3TX?>#NSMy1#G$0~ZiZ5W)?FBM4Uz&LG@DID~Ks;S|CxgkuQT5Y8doLpX?V5#c1lO@yNeR}s!4+(kHya2eq=!fk}(2-gwLBiu(gkZ>X4M8b`PBMDa$',
'&LrGPIFxWH;Z(w{gkuTU63!*uOE{QtG2vvw&4i-~R};=A+)X%~aJdjpC*01!@r3IM=M(N1!U2T~8aSbFL*a<R6@@bjcN7jOTv9lta7*Es!Zn3+3ilKaDqK`Jsc=)_sKQl+vkG?=4l7(%IIVD7;kd$eh4Tvc6%H(1SU9n8W8uicm4!14cNPvUTv|A_aBJb%!nK8S3-=ZdE?itVxo~sg=)%>7vkP|@4li6@IK6Ot;rPP!h4Tyd7Y{(Z0D~tW-av>)AYOrZ2I3uvhag^pcnabzh{qsagLn?&J%|S(UW9lO;!TK0Azp=e7UErqhap~ucpBnuh{qvbhj<?1eTWAlUWj-i;*E$$B3_AjCgPolhaz5zcq-zph{qyci@|dd?<K^85idqO8S!SsqY<x0JR9+D#KRFUM?4+zcEsZmuSYx|@qWYu5-&(RA@PR9BNDGjJR|Xr#6uD<NjxR-mc(NcuSq;7@t(wk5--Z&Nr^WV;!%lLC7zXdSK?ubmnEK-cw6FeiPt5bmv~>|fr%F;o|t%J;*p70CZ3siXX2rWmnNQ?cx&RZiPt8cn|N>H!HE|qo}74d;?WtrI`Qm6ygTvm#LE*;PrN<x_{8fI&riHR@c_jO6i-mRLGcL1D-_RAyhHI2#Y+@VQM^U*7{zN8&r!Tb@gT*E6i-sTN%1Jfs}#>tyi0?JDPCrXrzzg1c%0&Oisvcbr+A>^g^DLC-l%w_;+2YLD&DDhsN$uHrz+m6c&y^Jisvfct9Y>D#fm2@-mG}E;?;_0E8eYmxZ>rCrz_sB!Q&OLH^lQ5?^irv@q)z@7H?QQV)2T_GZybyJY?~b#Zwk<Sv+R(n#FS#?^!%(@uJ0(7H?WSYVoSYvlj1KJZ$l@#nTpVTRd*@y2bMr?^`@@@xsLu7jIlVa)Vbco_UCOE*`pg>Efx2w=N#Lc<ti3i}x-bym;~A$%{8H9=&+=;@OLLFCM;l`Qqt|w=W*Qc>Ut}i}$ZS0QCi^Pe6SG>LXBJf%**8cc4B5^(ClJ',
'L46DAV^Cj%`W)2vpgsuoMW|0geG}@VP+x`mER4Pj^<jkiGSsJ`z76$psINnP9_ss0ABg%w)F-085%rO%uS9(&>N`;%iuzL2r=q?U^|7e0MSU*ndr=>Z`eM{4qrMsS(WtLReKzX5Q6G-_a@41zz8&@PsINzTKI;2XACUTj)F-6AA@vcduSk7H>N`>&lKPU=r=-3m^)achNqtW0dr}{i`l8e)rM@ZkQK_#=eOBtbQXiK3vec)gzAg1}sjo|YUh4Z&ADH^W)F-CCG4+wDuS|Vr>N`^(n)=ezr>4F&^|7h1O?__ads82r`r_0lr@lG$(W$RaeRk@*Qy-rC^3<oNzP(T%pV8N+KEF`kpZWmR7pOi#^$lvSV9=N=7r&PBg)#5$BIjczZD&0)Y0@3a78xkB@I2Gt<K&c!I1edU=OpK{eeh7u<J`1BA<n__ALpDce_hV^^8L;EY`)r@JLWxN%u~zDb5Rqgb8cS3>#UTOIWNvL-I$AZ-I$A(iudE@$0vOn$$3v7*BN5~&+t0mKjplxckw)*^__+D|K2(|zbVenc}FpR-tU_j&Ody8gY%(Y?c-d<m$MgfE?GCu^?Jd%8cEMRyq{<LI4|vKzkHlOmW2l!KRU~~O%MBW-qnLJ&U3oAf^!(}7UX=c+xs{_>Q-ssZAmwFFXMc%+??}tbsp#EUG|aBb*ZhDbHw%nA4s|o#yNZEnsctynPuBKKkWWi&JRn&`A#S6a2-wzMIAlqnDM@&M>tRI@J;H+!PlJEwco?}S9^mSj46?v^SH~)h2IhPg!923wB&rS)B89dY$NBYZJ5A0W9wRTZsMB!oR_ug6X%AlI9QQ$!G`gEmlWaLzJ*VD-ScPidgqqn9JX1v25}zPAkIts#np!Mz!vejOwP`^gcH{DetybD{T{P|^TI}b0X#No#8|#>*h}iekb0auIcO8-e+@A4`o3$*d2GE`@&9`KOTFoqKL=#z?$rAZE57F(u*=k+Rsqg6YmtldDZj0SbJmlZbmhKlI1;!=',
'QoWhf|JutrpRdNcUm&Y(<XpMR8#uSE!YbZJ*?F9&R%!zAgW~--cd<y*>zoJ1`A|vuqc|@wH|HlM<v753iCHJ||1vkAzGR5z+_W^8I5#cjRL*USuEceZjN@Fruq9k?2R}Z^@|-wG(obtRA1qS_&Ij9tIcq_8RqW^B`@iuy!MJZ<^W5ZHSg+u}&4KR)pIv@Rvr>~d%KZaa@8F$W{M;KeI0x&>$2Oc7*7PpNQtreYx!_+e;Kjj{ZGpE2k8MrHxnUmQo52H>VUL2pjmNpz!JU8b{N}B}!nB4Zfg=Ui^%_82XED|%xcsz@bHrje-)un&JjdXiqI}<s2E31H-GO%nC;f;y55b?NgWnJwy@bCXu@?3#ICKl}-r%5Z*w4X!+oCyNY$Mh`*nJh}m35xabEn%+!nyFl*53~y4%HrbXYiXE=Q(dIAMdMf7<doC8ZZ9h*v;cV6TeyryfRp}H|MXFEH{ty$DFWZ!LP2so(J<TYC_zrE#IFt6Xsb2Gu{Qz8%#@FFqpCd?0GQCLf&WiZ{>J?b{EV630k5sFCY*&IF(~QJ96%r+lJ>Hc()$*JP=bI_B8PC6YST(vk`YW4m1UD&A{FDIA1?-vkderaODj2G4OXY@<h)(296py)$<7Pyf}XE;Xzo3z`jq|r-9vrPja5wJ1@tJ_Jv*q*8T&#8CcbZc;M2@@I!%xU-KOCIln<~12b~;<D9cunBx+d;KSUuz}RlEPk|pVfPWDfR_tfujSKL+0^cQqPZQ|XIyuiFKgjuJ9n<pXHiLna2AW^QT%N!;c{p#a!5HXqp!N;UZ>wGa_;sN281M`N<*va01xn@M^%Ncaj&ssZ!oLLarkKxpY3(uhBJkxF=y4!@JnVHKbydzyi{^ZlKm^a34>&V&j+><m<_P)&n=p^ppYRU&qyJM$%op*$9fAAxzdi=M-T%UheeHiz3%I8L{<QVPY5xXa$A2vvd=3A_rqE~q*#%j6e*5ip3=1-HUfSVy{J(vxuz&r#p2AM}x8;Jp@o(&T',
'hj{Q-e$R@xIH%peq&RpO{`tduaX#Bo%u(_GVu3yJPp-{*Z9mOC%z16MFptmwLzb?b+t!oU-+y;w&Tk9Ap8LDk=6ttK^AX?p+dhLH`CAs_dG$@lm*QNv>(?08<l%MJ7|Hvqa)sAbAs-B-zw}SUv5VcuIsX1aWxx;d=Uc$*%NY;5?a$hh_;<!5oEMjt;eh{(sleI&$v&RqT(}NA58rkXycfTzGOowx<6L2%Cok63_hA)y7{0iSJeU8~qB}fyKYc;Yk6VWL#&<t^M~=na0$zacT4{dH#XqqheCJwnPTc9ZiJUJtndkc-&B1g151a(w#P?fw*e~DCWaI^GIe`7@+rat6zBTFLAABn=f#={`GL@gRpcdvf_~wLz7vY<J8S#N{>Qc_3o7g*!bLuK#fB8nG1z*B9{3GHv-}mIF_y!z-efRa<!29aC5WEdv*I#aMp50{5rE5Eh^9ox2jCj}g?R?I;YrJkW#{eJN#_;q8;uT+YH~g@#Y9_=nzKXT6?!K}kFt^E9ayRGT6@3T2@f9iwd+5tQmVAiZSKyz0Ir4GcvrOi?WQ^gur)vqGnlIHk%o*~fs17^fOLhuzfG@m$9?sLd%j>WX#9SGlDLLW?uYV6^Vn_HhZ+t2|KktXjLpgVE9^x`@jPdtZB{)CtMKt;7Pe0*)ypNt>UA^~j^6z)9;l8{#@4&u#uf2qxdarOElJ{aZ&h<NA7kcbHGwN^hNDg8>y?-Quhv7ZmfP9pr>&Zts?7}|r9vFOx^Zs68-+A}+<n``+5B>3OpID7^C$h(I9^g65zwmAt1wMs$T{(W<8Z-8vchz;+EANV3@H5_Ji^1dZF8PJ`y=WTzf_K4OzJK0E@H4z~&LCd(&iXW%yqE&K&*}X*pYZ3S@Jrrl>0p1oQ^vqPdMAC1<~+hdnCIagALN|EpBBKbddG3Dig!#b;t22PE#%jX8oGevq#MGYdw(p!>l~4L8|ND4hX3;pFHSzsu!hJlyhDd#A9#ms<6On>-*K+t_tmh!yn}x|',
'Ng0i~-QGcrL%f61a-9e6<nITz96*^Id^7KWYo{nTY^Izxg>vK&O3u&p4w%mMAFyo=<$XTqff<)k_D)85raHg36z_M?Z+&=O)qbITe2>>XvmfuLVMpFy=5v$*<G$m1avk1Y=K5Gta=miZ;rg}W=Z#vyxg?|AoaZ>U9oKi<O{SIOJHdW=Cj>|Gxo!QO&#~_XKG#a{Q{G>)@8G<Xbi1ezSsze8%Gag7^yK|5*if1Jlt8^++O8(&R9>$|J#3Ht?ETfdhkCl1*SoPV^=3=871ZwnT<2YRz$^0pR+{rE_qRs8<2^Ks&*$h}?yD1J5NCK#ZFX_L6>h=(cZT|LX~<sg(^BxC-dmBJqj}fO`+fKk@t*gYeLlyVf5rQG)17lU-|nN`_>=+hk2hg1cv9X#R?Z7G|3SOy_yKz4i>SdlqtUr(M^k6w{Lu_~=qIw&;_q{gjNyEkzp%f2h4WPB{Lp2bH(D;YhjT^$#6I)Y7}<wBvs&=$z9xB~=e`zS;(C1T@*_|3b*Tq`>Ff0q;sIa(OYb?yv>5qlBex)a^^GrAl>X!)d1<p2m*X7NUi2p`8Y3?7t*cMJv$X?wy}mt@Ij`o>pZxtF8MzK;C-J_nIO#WTuPjf0S|4$b?`1TeukY>ab@az~|K%LlC-m3OAndt6x+vy>`O}SH9Fq0U2gU`B7_St)_ZN9`qY;<;zb=Km(O=&_n7p~yoFm)rAI`h&?!k5Y2Yl&aoYVF`<DZ?3M}EneoAJ_8<fZ<VMHzo=d`Q0C?$zK4`j7m;Il6!L<ohmv$LqX1nep871B`3pf|vv3Pi)P2)p~(4`M2;#fpi}jZ{}E6nB}1Fa2<iN^~v9>UIls_XjBJvgh1;axb8r=rJU<K;2*}rBfrKv1SaoC-XEA-1@TT`#RKrg1Dj_t57^g|^N#;0L0;g+TpyTE6ks0kqGleJsfNH$1cC=RuR0>?J@cF&nFr?ZF^?)T^D6VG(#*T+eB91F>?G$kcUe!K;DF_@W5H1ynCDG9&3Qo!<Et^R',
'EXR4#yJz$Nj`^6U{yi*?bFrNbIX8O~)+HF<j`_bWC-gljg`f3>EdO#Ic){n0JCe$OWPPP>O4cD-R%acq`{bjXYkrS;`}nHR|D@UbS;tt#d2C7BZ?KMVbQJ3u7fLl?Jt}yWb+22j4<sDh&3fj3p6iz880Qt`x=Fra2|w$X)r<1?O^4)R{rNAR!!{`I0oJz{@jSZOsd?_{s#QE6Zf9AZpLp!?I-aLGkKgyG8PB(So0FdxOu{@3bMhDDA%1z6I87n8O`0qH3tY?GFsT@EkL;*toBKB*&N1faTRac<GVzmDS^4{&KjOSU^YKSK*Y--o0pu_K1zf=Vp-xQ$HzF>UvKQ*MmYmKNEYof$AF)O|o-5ou73$%Z?$?N84O<Fa#4@E9>h+ezmH9cFGlNHBIcx@wYdQZ0IJxEiec&aQ*lVa;TY{H~7e-wpuQBUg)OW2#{{@fFS}hSczV+Lv0CCCuJSTc6=fPMf_a%<FXihmE=XiJ%@lnpNv0g|EJkR=|4xW!SZX|FXYtrs}#A#oHS7pnd1LqXlinjwkV5`1p5|59(j{35#2aip(4X=m$vrV0h`n+w)A6JM=2Y}n!j+O#WV!POH5OMCcz{zay?trgkw?xz>Ze9+!h&^Wy@)k?Z23~Bhd4M>2i@V^J*n1`dFS3tF1w7FHbAFyHzN|cWul8-V@m%b`HzE)6Qe&LUYJXgh{DJpX02n$fg`gjflxct?Ida6~Tu?`eOVCG0jaARdm*`EtMvwgbdDt`Z8K=x4&tq{Jp3lDNI-ZB)P!H&t<J>#&s~q?G0e^A4c?$lN!(Vqcd5IhTBJU&-_`b7XW#B%}io=12I2)`Zzop%2@>jmQgXiZQ^$>FpoYQYblV5X!Jj89wfj>Eq_U5_uf9J(|IPc#kZ!l&!@D!)tMP6FOlr`i9CXip0zXkARSK0ME*T42N>~B}g<XCrCkG$mF4K0G_=bD%s&&@S24ExXZ>mABn3&=}5UYR_oi`TJkt_RI=-n#1z`Lix>ChXrZR|nwH',
'VQFUIy25g908b~Z*k-I}Sd~S<iNhN9$N7$7ZHqv!!g}2W?=x&zU*Mr(6JCO67B;6o_Ho$Ch5Y}m=YhM29sC#gP1v8WflG#6y9Pd5*yFXFj}_YndK;FQ2plxrF$#EA_!lprH{n@I0v`)6&<pnyUV1#|WqmykcyM^5uE5E{+Z4d{hj+gTyf1uEcYe;O+nkp*wJ`A4@OeF9cfwar0In9kc|7Mo?Cl2LO8D`7;OB&&y99ni_>Jb!tMEt1F~=wTbvX1j{8K5;<qCd{`-pHA1<o9iA_+KTM8@sVyNFyB@cj{mws4+SX)otvRLzTZi>O-?e29pq*|F~<+Qh+*MRZ*RoGjwIY@EX}WGd`k#HdT)J4H<V$T=F*-8f%0V*U;AoFbOb1#c)~T_Nn3h^^~^M@H=N;r!T$!-b$P5r0$!-WqW>AD(l>rT4)1B5p3g^N4tmj_}*FA;7^RUTuf{i+Fny`zONv8?Vndj&n@R`6wM*an5LDWCVXsQ5`(r$TY36-jNyW0AG*HnhN?Fne!lU;K+Psu@55)&IMi_S@aC>+sKl)fQLqwJr2Atvf>2rZ6d2?2aX+CeKGLZ$Xai3ossp@;QAsPX2SE2Z0w_Kx(9sF$mTV{kBMx#3wjdS+QaW}mj?bJvO{w4OCmcy0-qtW^EBX)kzJ$kTq3*m1%4gbeGTWQ^w<M_Vr0){(9_6X?H^KlflEgA?w3e;<PhbvCI2n;d*gljKYflqSKp)W)ph8)be+0x-G}Z=_o@5VI%r+APFgpuqt;dHtaaCO&~wpq(sR>u)N|Ey)^nF0NH3%((i`cKp;ywgklslTrI*rE>8<owdM!Pd-fJIdUud6b-)J9cUumCd-)SFeUuvIf-)bLgUu&Oh-^&ijF33*EZpe-pc13n3WOrnTWS3;8WVd9;WY=WpWcOqTWfx^9WjAF<WmjcqWp`zVWtU~AWw#AGF1sGG^RoN$1M&;<6Y?AKBl0WqGx9s~L-I@VQ}SE#WAbbAbB5oO9}M|L`APXr`BC{*',
'`C0j0`C<8G`DyuW`EmJm`FZ($BMwkp5Q-BNHz<x!T%kBaafjj%#U+YU6t^gjQCy=qM{$o42PrNJ#Yu{r6h|qpQk<o@OL3UuGR0|%+Z4wsu2Y<+xX*|K6&Hr$M8%DYBNbOF&Q#o~I8<?|;#9@0ienYmD$X_HUd6$oxL9$r;%3FsimMf8EACbtuDD!ry5e@l@kU&)I6oBkD-Td!pgci&gYpRF70NS|cPI~0UZOljd5e+9D6a|SIm&yK2PrR7o}|1<d6e=h<yp$Rl!qxVQ=X>0&B)`F*M;&t<$cNnl@}^cRNkmOQhBBFOy!-*LzR~*Pc`yZ<*}i>R(Y=SUgg2ci<KuUZ&n_yyjpp-@^0nf%FC6fD{udwJYISIf93hg`&9=}T|jjL)eTffP+dWF2Gt!@hfrNYbqdujRL4+V!>Dtp?h&eks4k*9iRvb*qo}T;I*aNqs>7%*qdJZ1Hmc*OuA@4S>OQIi8FeAmi9&TF)sa+JQk_Y4C)J@;mr|Wdbt~1eRM%3SOLZ^R!BiJholJEz)zMT}Q=Lt9H=_=xx?HGEr@Ec$c&h8E&ZoMc>VT>Xs!pi7q3VdLE2_?@x})lls!OU)sk)`=n5t{4&Z)Ym>Y%ENs!nRuO;tw?)m2qzRozu}Sk+}!r&ZlnbzIeTRp(XRS9M_3g;gh3-B@*G)s<CeR^3^3Xw{`vr&irsb!^qORp(aSTXk^N#Z@O)-CT8a)zwvJSKVE8c-7@qr&rzHsN<`yAFA`K?k^lbxPWj1;ReDHgewSV5bhuxLb!x*3gH&QF@$Rf=Me5897MQ?a1!Ar!cl~)2xk%QA{<7zjBpy^Ho|d)>j>u&?jsyXxR7un;YPxdgewVW67D1%O1PA8D&bbbv4m?0=MwHE989>Fa5CX$!qJ4Q31<`T7Q*2STuwM$2)7fCCtOcBpKw3nfCerooG^qN8aSeGMd6IX9fd;*mlRGZ+)_BEa82Qy!aaq93KtblD%?~!s&G}|tioM|!wQ!bPAlA2IIeJA;k?3q',
'g#!y07EUbOSU9q9W#P=iorOaSmljSf+*&xcaBbn-!o7uq3l|qoF5Fx=x^Q*j?84oJ!wZ)ePA}YEIKFUw;rzn=#RCv8Ks*8Q2E-!}uRuHl@eag8Fn9^#DTH_n;xUNVAfAJG58^?H7a^X6coX7Lh*u$=g?Ja@VThL@o`!fE;&F)AA)beLAL4<C7b2dBcq8JGh*u(>iFhaCp@^3vo{D%Y;<1R=BA$zQFXF+77bBjGcr)VBh*u+?jd(ZW;fR+bo{o4s;_-;rBc6|VKjHz27bKpLcthe5iB}|^k$6W24@tbF5Kl?GCGnWVYZA{%yeILX#ETM7O1vrYsKl!h&q};2@vy|p5>HFKE%CU->k`jPyf5*<#0wKoOuRAi$iyoX&rG~C@zBIe6HiUNHSyTQYZK2+yf^XS#ETP8PP{qs=)|iN&rZBM@$kgU6HiaPJ@NR&>l4q<;QfgQ7~%zrCn(;ac!c5=if1U^p?HYmC5opg-lBMn;x&rrDBh!Zkm5y(Cn?^fc$DH*if1X_rFfX)Ws0XM-llk*;&qDWDc+}epyGv!Co0~kc%<T$if1a`sd%X3rHZF2-l}-4;<bwBD&DJju;RsvCoA5pc(mfxif3!^ZpFh5@p8q}6>nEOUh#Ux^A+z`JYeyH#S<29SUh6!ip4V)?^rx!@sh<;7H?TRX7QTEa~AJeJZSNv#gi6qT0Cm;s>QPw?^--;@v_C!7H?ZTZt=Rs^A_)0JaF;C#S<5ATs(5|%EdDm?_4}|@zTXp7jIoWcJbQ9a~JPjJb3Zq#gi9rUOam7>cz7c?_NB7gO@L!eu%d(9=~|~;`xjBuRZ|v1*lIzeFN$vP+x)i4Agg^J_Pk8s82zC3+iJ~UxWG_)c2r12=zs%PeOeY>Z4F!h59ViccDHE^<}6}Lwy_S<4|9R`aIP4p*|4xg{V(NeIx24QD2GrOw@OxJ{0w(s82<GE9zrWUyJ%&)c2x381==dPey$+>Z4I#jrwfVccVTW_2sBfM}0f$<56Fa`g{)G',
'+-URxUHPxRAoU5UZ%BPa>MK&8k@}9*hortF^(m=uNqtP}Yf_(+`kvGWrM@WjNvUs2eN^hJQlFLjuGEL6zAW`=sc%btT<YsmpO^Z+)CZ=%F!hP4Z%ln;>MK*9nflJuho-(X^{J_EO?_<YYg3<_`rgzBr@lD#$*FHneRS%pQ=eU^@6PDMQ(s=FPtWMvQy*WbuTOn`>ibh4p!x#UCm8%sT{ijEU$)Qxznx&1WW4_O&;RQc_+Qdz$V5YW4Ee#3Zw>j%kWUPG*N~SCdD4)-8S+;{&M@RKL$)?#c|)c#?mxzmM-4g2kaZ2|FfRLmA-fpTM(JN_$Xv#48SCmZKHvL;aht~VCZ?oJsAb%CPvdzErF8c<6ps3v^pW;3>HRgzceKw*aXfD>DVF;p>CH@IxoA(4{@p|Q5A8$Jv)Yu84ZYyGk^i$14ZXQ-{Qr#slvlGGdb5J^+`p86##8=5{Y(122j#&(jK{$JowUQyyUny`NoyxluArSrT0|aP(rjbBr*R)7jW0|2BkfAk;IWi_Xh)MeSEp>1jIvQL${MuON#$6$Nh<u1GN-X#>5Th~jN)TW%0&5|`y%*^_9u9Qb~kvM{w}yLnsVJX%6ar7!STlX2JE10XZ*bm^*LCQb~u=|8)Z2CX5b_JOyGVK%2U*@z-Ic<z%1%xVCWZ=t(#F+<mU&nQm+E$sZ_hCw8Q=rBPmzXulPrCJ^ZbBef|>0@1v-HzL$nP#{K16Fq^Wkg|Z5tgD)kYllM98t#>caJM>PVfAKb<-Sd7~fa?9qMY)^)I&nmA$|{d3?ewb&XKA++cur<gLOt%wgs_X;7Jo8M@r>hjd&+ZP#D6SIxhXegyM~lev}^9;d|vK;+?Vc*jB7rfroZ{rkNWp1HT}`YgIu4FZS&Il_^Br!mhkr<N^#xaUmi@!I)2i-KoQDWX(_W(kKS%)T=2FWKPT=a?R;Ea;v}&bX=h>^Q!ipJ@pEJ9(Qm#vO+9>5!RS}s$$j@aJMGS^+0@^ce%{Z^ZuDROo<Bxe',
'=y#g=8ArJN-|+hXslk2rd>QxSGuA1So>k#<dpe!=;mNJcl$p7nk2{{CUtY!Qe|Vp9#KY9Ivk&UhPTwE7h2~^4?ZVv~s3RoZ@l#*#Wa0YWu0(r$s}=q5&4G*qZ%m|pzCIu4h$dZIWtgFj{Jtw&sh^j(@_Am`Kzn>~CF9D!=g{w67|ZoK-={OpRukHfv&H#;XQGGGjJ;yKbNVQMJ~f?j$Vr|Ho%F|-e169tGtNA=oO=3uvy6<VT#Tm=@1vd^>X@Ep(8hJ%zlL%6zDhM{7OyeB-P47BX;&QW*3R$wx!XVS^S1S;UD)#cHO*%;?$eFO_#8Im<a)20K>N7nA??7gtbZh}8aIJv_8k4n@??yom)2!GzxXFUhlP8VGhTg^j^~ux=qKjnpnacJk^W&uW1d_3OFLfQw5~kAbV^soy_4EcWW4+h*Y~H&v_oTa(|?Y3%%eGf%zgU99`55|V|af<YS2#&wo!iv97SCqsb4?jkx6~i)1LG?LVwbuRac&~`j*$-c@+Pzqn+#4ZsIY++tWU`9EkiWso6d1QPawduNp6<U25=-&%0hl?!(&S>6dGqn9n>WfpKq@Jha~x8wTlS`f>l1p2Yhov5;|c(Ushfg;vr26<EyumUkNcO|HST*V&uCWZ3o<>O4sq1KeNf{^avXO<XJKi%NW7@;Jst5epdSxJvPTw!fL5ncLvG1p`m<eL-)>l*C_eKo5c+tFb=tc0Kdp7z^|7m(3`juk6Hd@(I>0_#pig=CO5xbc+LtvtF6WJpAu<#AVO!{(|muFZ@IB#LkV3rhYZ%$NUU`9o*Xo_Z{3-74|5&EsFJvjn@!A2iML(ohrDh3hY8~=_}S17EXoT3(n25k9q%A=Ks@*Fi)SjneQ8yk@bm@<9S`fp0I8)xKwS{8^&SZ1$&)@oe6gJF+cB^pX<}6(Q4K)`ao}kO-7;K7py;#b&FaPfR_ZT{lM$1*p0Y$nOeNh;>PtAjAMNu@1BgrpZmeD1;5O~x<I<WxE?9HA+HH0OXT&r',
'#`Aitw&JY&j9^{J^Dpa0AL_B57PEFJ@#>GnpP!V4J_PO!!9EMz*nsZ|T)KofC~!88`1C0Y>t4SnBTjuFc_Y@j?5xLa`2f2WSa$=@J+N{I>{DQ|QO}#puq!a#%=4Eg?cx88Z3BND_`%CM+u)fUShvdn`x)rI^d9SbDX~t0R^wW;4)_oJaG-u=-d~MbsQ(5k-6GyzCN<B0E?OHnQy~A~9IQvqMSL5`w0Se@mIqQ2CqK;3aqi=NnKrPVn>Yiwga1Qc*f;;1s;sj<k8H@g>P5r>{u`5sUtcVFkN7$5h5y)a#Bu%uk+ARnoeN;c{hQLV-o1Jz^ufQF@uYuFH#~p;v_G)U{_$zxr~E&5=HCaeg`f2IdB}Qt*QjBvTbI5-T)oAQ#MS$ue&??~DkJOR<B89g7z_L6|7r;9+BrM34xXtN>%pmVa86Q$2YHs?dMYjJ_fxQbzK_+}SNxjihWnnbCC+}g4(t3^{&_`Q{YT;&CtN=hcbGBFz$5s18zxjGe({Oth%ab+oOs9YWr>5NBW^#E@uY9?0qBRXcM$jE>s*02O{*cqiyCvzgRj;V+>ftPJl|g`72*kB!LJ??AFp(V_)%TrD_=Awt`pXnxRI$YaYj#N;#F~l@Z7xrq=kL;-t!?3@m{@)>-V1BbB4IvlrqGtTM#GRo}IY$+Lun^&YSWOKkP=_bZX|?#4E3?CjL2&c-?njp-$lKavk^SZ9R<mbCb02lipf;h-+7>k2;mN#1sC1fx);RZ*~j(hBw^|;=WO7iR0Us!~Q1vv!afe_+j-)hTB<rUi_nFxbMW9X;3Flyf72bCGn&k`kHuf1g<}EM=au=#P#hFw<j*U1izIyw-WR<aq3psp~SK2p<jtZCqe%c`+k5vCU$8LzmwSNEc7<9aS6nMi8WW^xg}PNz`7?EABy-iG2b)f$%$DT;W;LzISTubn5+QoOrmuu>_mb$62DJ)HyY>EB)s^5xF+F2Pw0KZwMV$WgmbNU-v03$c&-Wio1*TPu<bhPEeUJ+',
'{1TQt#C}Vd-2>~JF!>YW?S#=2fHx-$Nr$>$LZ2P*-wB=TA<ju?`3Cx$&}atY$b_1uIY+C)bHoh^#TW2=|NIS5=SavFhCDPO?K!MRLh{+b0TS%(aorw&5$KoaV;JHx&#Ra48=l7(5yyLO9fE)GT-=6u%X4}I?#FXvBd*)CXBX_HXY&cfnV!{mke7NECBp7_W@U%}@J#*&cG5HY7p#wG$VJ#e&v)rjNAPt04so!j&3Wu=Pt#&pA5Yz1f!}$mWk$Z`Df=tzfv0Fu*a1)8^U!xs)&W=tPr59~Z#+?V5kGmHtFW%|Ndr-ji1#$YK8Sx?2L2=d--7VV@sA2)pT^&+2)h}7sU`F{{;#oEpZH^Yfv3gqPr&-c@2rh|8ozNp;`sR0Um(7WU;GQ!D}GKYtWW&3Eu1GYz6qXR{747t((&J)gPz8Jw;XmWzWZ3p4ug>w$G05H_cfV{I3&LAR`{RzuOB0>i?5gqaanw+9}#cF7k-NU5TCCN_Gx_fe>j&S(@(I6@o9=QU>f=k>k}V#68k>hx(V^WJFpt@f!nhUd9VBZdBkh(H$nI*_lxS#BlqLwSP%ET2<$iajp>Nf-IsGBzj2?tg#GJ2J+mCs-L8oD+=m;&zPa}`MEv01)rE7>w$8wQc5k?ZxYWHS5A2eA#XRVddr5lKdEE1V<8{sMj{3BFdSN_Y_tYpn7xzTgOWotrqCV^%RT=r6`-joUhuuSN!k)MX)vC($`!ejQyU#$>`P@Cfg57g>Gx5GU#ll~>+ke8m7<b!L&<l5~hR{cM^EI$v?r&4T54f8wfgN-=stA2>*Z0BCy6ax!=hyxV&(B@+CgOH?4HM^)d|eZ9hr8+q_(gY>oQN~sm5%a$Dh@^-;jT~}cG6wG81&p-t^(J+Y!~R6yUd1>lrG@M?$VQZpQW=QZ*Z5o8%DV^7v-8alzV@qe9q@xy4V=X6`Y$=CNI~$%w>L0*@a8^{iFDN%a8k&*SD6>q2j+>*UGi|{wilVZ>3r<{=Rx%{(nv9M6Qb?',
'CD$ou8?IYt#E<TV$9X?ZD)D_y@2}^&FX45z`hjy^+79RXcUXvc(%tzMpJVqjuy5{O2jM^5-?agL>>ikTJN3cLzyDyLLVd~4bssmdKK1D~pU1RLh+Et<BltOUZz7L#FZu)a-o5;CTgYFNsE->PA<uPhI{~}m-qQ^G+<h<{d7}IG$FLvnQ@1&<=G=Sg)0Ki;hg(yDr@J4z;HTWrH&VXo#r^Q10rf8N8{|dtrr}tZc<1rw+~4_WztZf3p2UCIg?27?F+BJ9LfJ>qZj^vOi?7s!^K5D!p`B|~9Q!N2#d*a4@g0phIK4XH{^JL=LA@{j$KkX)6ZRwDjh~Sgc0GRKBHEeN<>0U5HwO^6#{c#N*B}4;3+Q$H*+|q8<F7TP9eQvW_Zk24Yxsls4>8c!_`qJoIUd&v@M1hEH{$+0UtWd&dh%z6-Sw25jQq({H3i>S|1jb=Ps@q?xodyeRnLIo$eTPPS0NtoOnHg8+cUq>DEhA(*pHs=Be0J>hbzGzdCp{o{qo$%2L1Frt%7y)yd6E5e)KM`BO$!0hyL~rt|uYas$2BSgX#CHc6mhq&3rYX-O`Kn^Utu~5=PWQ9V21N8Q?Mr3%k*uum2L)mGIj~@P86c{)6YAaP>X%u!N@>I0xu`d&C`yrX!r+6I}{*iNq}D8TS+(f&H9V`D@&FVxv639}_#|MBbg)zdWuxaZF#--4bW+!@fyel^)lVxN{z^Kk-C9!~uy{Zo-}?K3xhspZIY&d1bb~i0{3re?UC%&Akl#2XE<HjC<?kC4Z~+EchR9-{j;Sj{23nx0&^Ezuwgm<eTk&e3X37b2G^!I}82vzPwM~rO&#7yspa33$o0}YVfjll6TdeJf!AtGm$4dtsmq6GRU`lGd!qc_*S1COFq+9*frl@tL@A`)*~MC#UANSp3zh0U#Zd~zV+wnu$}zky;I2l$v~b<C*tk?At{)5P5Bk^fqz+5${n%f7oFVrCHXY{yD<;^nt5DOPShs@DWl2j$P>mqx_q)*<ZEYU',
'KHH@<^Vnf+nV0|kOCRRRr%|^J{N`dlex@1o-uqkd+yn2!fQJX|EPDmhKSbOfEYys7f3<6{2f-FC5w8XNJcm6CjvdOn!u;f~h|+B%-{AMg(2wBNMDis5-O!8tfL5%trHEqvGuJuRy~<7ANtmQI>yKR{i6;%e%DU0?CB)}{?aDgwe#S9L=L4)~K0ZfW(!J>Je@Xpb@6-S3bM(3T9(}K_L)WG2G_G6s5xOtkr|w(ppmotYY2CDrT34;J*4=mxdM=^or01sRsOPHZtmiI0kX}enq&J2hNv}eBCcTp$N-w3S(p%}V^jdl@z1Kd_zA*NQ_DyIXX<uodY2RrdYF}!fYTs%fYhP=h8~a{%AY>P0CuBEdM`TxIXJmI|hh&#zr)0Ne$7I(GJ14srvV*dVvXio#vZJ!Ava_<gvcs~=veUBLvg5MrhMkw)5BUN41^Eg24fzrI75N$Y9r+>oCHX1&E%`C|HTgNi@5v8_{G$A%{HFY<{Hpw{{I2}4{IdMC{I>kK{JQ+S{J!D<#RZBJ6gL=ggyM=&oT0cwafsp)#VLwg6vrs8QJkZ=M{$tiBE?CHn-oVWu2P((xJz-E;xff)irW;&DXvqTr?^jXpb-};P7KA3iX#<QD$Z2gsW?<|sp3?{t%_q6*DB6c+^aZPak1iL#m$PN6;~_HR@|*PTyeSLbj9t8;}zE{&R5*8JV1GY@&x4#$|IClD9=#dp*%!+iSiWXEy`n**C@|X-lIH7d6DuY<xR??lvgRwQr@LJER>fiPgCAz<Z;UDl;<h$Qy!?iP<f*AM&*&pE0t#|?^GVDyi|Fr@>b=s%4?P9D(_Vuth`uxvhrr-(aNipXDjbk9<IDxdAjm;<?+hvmFFw(R~<lg0o4gqH&7ixbp_QKRCiDvLUjq%DO9&m9Yb{u)j3r6P#r{d5!FdlH&Go$brscFRCh7zFsjRh>NKj`sE(t$j_N$B`=}12x{&HbsvD_}q`H#oOsYGn4yC%3>Qt&*sg9+(mg-!pd#Mhlx|r%@',
's+*~frn;KyY^uAd4yU@D>U65x8Ff6>^+I(%)%{clR9#SYLe&jbM^s%=bw<@4RfkkvQguqzEmg-<T~l>V)jd@QRb5ndQq@gWM^#-_byn3~Rfjd|vZ~XD>b9!ms;;X#uj;<41FJ5qI<e};sw1netU9ym&Z<MJF0DGX>ei}btFEm&x9Z-igR3sCI=SlRMjc&s^-!H%b$8X_RhL(tUUhrb@m1GXonLi-;Q+z~gcAri5RM>RK{$hO2jLLHC4^H5w-AmYTthgAfqMuC3E?8bNramSM-i?foJF{ca2Vk-!fAxt2*(kwBb-OLk8mL2Lc)oJ8wp1et|XjExRY=w;Zg=pCEO~6V+q$1&L!MSIGAuT;bg+igrf;p6V4{wO*ouzIpK7|?S$hA*Avbs+)p^5a6#dO!VQHZ3Re`)DBRJ&A%#nZa7y8p!ZC$w3g;B=DI8R|sBlu@rovH$s|sfo?kXHsxU6tm;kLqYh3g9E749n>Sh%ooV&TTZk%cP@XBO@(9NNI8g;R%cYvI_!wS{vF_ZAK=TwFN0aC714!qtVd3wIX|FI-+Yy>NTs_`>yt^9%PE4?w&C@dU&h5RX8-0`UyQI}i^+yae$S#9I)LLA(a>9K?GN4??^MgC`;0M2JTrUWIrT;$4V`Azp@f8scq;$01&acpl<?hzBBGh<GC6jfh7gUWs@n;+=?xB3_DkD&nn($0A;fcrN0-hzBEHjCeBQ&4@=MUX6G*;@yacBVLYpI^yk!$0J^kcs}C&hzBHIka$Ak4T(o2UXgf4;vI>HBwmtuN(OI9Jf;w@NjxX<p2UL^FG@Tq@utM160b@;EAg(x!xAq`JT39I#N!gLOFS>}zQh9)FHAf!@y5g>6R%7>Gx5&ELlZAeJT>vw#A6e$O*}X8-o%3wFHSr;@#e&%6R%D@JMr$s!xJx0JU#LD#N!jMPdq>I{=@?mFHk%|@dm{s6t7S`L-7v9LliGjJVo&q#bXq&Q9MWS9>s$cFH$^7@g~Ki6t7Y|OYttn!xS%5',
'JWcU7#p4vOQ#?=cKE(qSFH}5H@kYfX6|Yo0Q}Is4LlrMoJXP^l#bXt(RXkVmUd4kIFIGHR@n*%N6|Yu2Tk&qi!xb-AJYDg2#p4yPS3F<we#HY8FIYTb@rK1C7Oz-5WATo~Ll!StJZ15g#bXw)Sv+U)p2dR}FIqfl@utP27Oz@7Yw@nd!xk@FJZ<r|#p4d~x(%MUc;DiIix(c^i5t9e@yNw17tcJzJ2!af;-!ap>IQFJJoXT;T|9S#_bwiMh!=10<i(p0@#w{?H+c5q-G_Mi;^m8{Z}9fT;}7xr#q$^M-{=ESUqGl&Kz#%1BT!#~(PyB(gHRuW`V!QqpuPq5F&KRf>T?M7J*W>teG%%DP~U|5DAZS>J`1DoLVXyaz6|wgsBc4k9O~;(pNINB)CZ!z5cP?uZ$y10>MJq&Ow@N0>O)aqiuzR4x1v54^|h$aMSU;ogHd0M`ef8Mqdpq-)u_)#eK+dEQD2Vwbkw(_J|6Y;sLw}zKk5TgUy%BQ)HkF)BJ~xi&q#eo>O)drlKPa?x1>HM^);!_NqtZ1gHm6V`lQr1r9LY4RjJQPeOKzkQeT$(wA8nyJ}&ijsn09a_hs~fsV^+lC#JqJ^^qBUW$H5v^_{5?O?_$VQ&ZoX`q<Rhram|Iy{QjQeR1lOQ{SBW=+sxIKD+<je;0cFpZfUzFZuuzt@pn*eSUp@eSUp@eSUp@eSUp@eSUp@eSUp@eSUp@eSUp@eSUp@eSUp@{Xh9~JjL7;Q`!NT?_tWDWh!Ku2bg1Fs<RpMeN3$<W6p%B&mhbbF^%ktc`~Ny-(zl)X~ndykUO?xu7&CNW6UElUCo*j@>yTZV={gE69a_Ij;}D^#+-H~=D(QpW#&AKavN_$)~`GUvfY!Fko{+PA;;CK4>>oy8RXhqn7d=%w*zw}%x7kE-o@RK4IpC%V1A1^*bnoZEYU+S-^G%BLOaM3OEI6wQu8q8-&k5c#rzaYpJ>cWvW%*U`52a&<J&;4{uA?DEWbtLzAS%s!2BD_',
'?cXq0$MPl$^EWJk{zd*Htx?zhfXrS2^ERv{Hs^+{l@Z@>Z8fVXWM5Yk$kF35C&fD3Tm^E?xc-p)tlJ^aOvXA}?<K>$8Efp^_5YDJQwG$rZD0I~c~Q1p`8ijm^ltvWZdv}l?eU)>``5s{5!<-4m<M8;S08h9Z0j#z{)X*f9nAT#oj-;7HnxWqx<S6%iFqA%Yqr{usiz-<%;U!U?d4iwE|I<deypdxLloxa*avq02Xg$5qmc9CvqEkxiS@M~>W8@__6sX97smeZ4Cd(A-@U^9JFI5T$4Q-HA7q|%)ga5J#=IFvJ=ZYEw(qh2j((Rgr^PXL4d%o+=JdfFF2~xun2Y7udk6aA_-hpAFgfm|#(WaT>s9>O|K+OxNN2JcxL)U%@18*xsgC=1R-KA@RL*aXW3G#{%L~juaeg1fyc*{uJFd^Uz>oP)&J9nouFeDd%RrtTfpvG@{R(q@oUbq7xjB8Uv7cNKH!%0am8k;m+x69K{=UN1WRUfP&;wVyG%q2)%YgY{u2C-RSJy93F@MapY#r9mwe_28kcVGkU0mmTSA@KK4Rd8&uXACpmMgJi=Kshr*H760VQGKiyr|rx@%dpTTjYbRo(B3D_RSH>PPMUqVf_z1h8+C`o=4a(jW9PXY{@|E`>>6}Fux~kU*~TiPZj0oU3qpA^3e}ihp?EB*vDbsM%YK;&gr<X@KoDjC&IIF%uINp1(zVpx5IM?uWjQTuWx^WJqz#X_CbDE8TuRkLr=`73ZFO#bJW6Tw}BlCU!DVViNZIYy9xPQV?X4vy;zs<a|!su@SEwepTnP|!`!m)H*VZlc>H$Ehl?=R;2g7vgX1C7ghB5kvX;WUsEGWPG3O|vL{82%toRS-uGJdX5wdZ@UC7q;FxM!e%W&vjMBj0k#}zT8%Ld5NS!Y5{+Rx9KnHP2=V&P!SkBeBjFfZhWd3e4NJNm%xMeNUj^^N#_4W56*pEj&R#NQQRM<cE`g1v~iR~Gv|;)xITH{#`d&UJh1z<Nfw8)FVt',
'gm1tv|B;dAp0LZ2&eC{Zk&zElLw?Z;dJ&m+8|-Rirdtyrv)wKTnR^G;Ke9ku=v!pr$5|jtlsy7jrWfpdWQBgXzQ`(dvEGqi`}p~_#_fcx`vCSevSAeFx<xig!})hjKk)yWuZ10mY*nBXWShyD_Z8XhKmo`OzhiwOJ1vG?i|kT|vg<|I=g97P;C~`}G{-!s$eyjS4<maO<9+vf1b-IUyHgs<{mUqyKKgH|-y84K|LJq|x%wV`udYMarR&sn>ppZ}x=-D=)<Nr{b<(<N9ks4nXRW)QgPx0?lb)NNqn@jtv!1*3KzboPG4w`y6w)i{ne<M2D7}=PN^hmd(rf9t^j`Zw`$GFf`$qdn`%3#v`%e2%`%?Q<`&Ro{`&#>4`(AcHcEPX{vKt{gBD*3xBfBFzB)cR#CA%d%Cc7p(C%Y#*D7z>-DZ42<D!VE>YuH`c;gDUHotE8}9hY5~otNF0ACO;=pOD{>ACX^?pON2@A2R%s{8Y$q$&bme$<N8}$q&je%1_E~%8$yg%FoL0$`2cUS$;a?x8=v>*X8Hs_Z0^yE>N7HxIuA*;tItXiaU%rL~%(dPEp*VI7V@e;vB_2ih~pvDNa({q&P}(mEtTT?ou2UipvzIDQ;67r?^gWp5i{mfr<+iCn|1K9BIUriZerTr{Yk>rHWG(w<?ZRT&p-&aj)WF#l?z~jksBHbSSP?oUOQ9ak%1g#p#OM6~`;CSDdf7-^c@$7liTz<qgUslvgOvP~M?DM0ttw6y+_-W0cnzd5-d)P#&bbNO_X-CgoAetCVLc?@}J7yi9qT@;2phMqZ~pFO>Hw4^&>LJW+Y0@<`>C$}^RBDi2j&sytPBtMb^-ug|Z~ug|Z~ug|Z~ug|Z~ug|Z~ug|Z~ug|Z~ug|Z~ug|Z~ug|alw_aiEUQ9b=^iggy`Z|sNOwBLUJVVVl)VxE@KQ!hcYCd9UUZOESQS%f-^A(MGi<-Zvd5oIRsCkW=->7+xn(wH2kDC9ed61e9sd<r_AE|kgnlGt&',
'lbS#IE}s3_?<~eVG3R@>!`j)08{=mmw&q)E-lgVWzMQ>~?X|{yOwG%D?qNUmvwggfr~TNku6dh}&ayxEVPDSEd=SRI{(CEUU3Uxeb8hcr|Lv{P?AyJuoA1A#oA0|ikA2*ieLP3vQXBT2{=Jvyaa;&%#CGE5yuLHb_<aAo&*yqN4g2m-*1<W4NhgN#I*%>mxdum%@Z5mIH`zyg&{+Td9zM^#MnCQzo|l-k%ga9H9dYb8-u8ff^;=H!Jcvyjs23Z?vtM>yYxYgA$<O()t3L6(j1>oPzGc#~VVr-vq%f~%;ZuIz{25%Ixh41Ve2kmyr=Bs0{q?^%*{?lyA?M^y&PIKju#V>){FIaHHfA~d=trgCIS?a8v+sJ?zomG-htaP-Xamm?=pW>~-S3+4youf`*zerqANIR<%ftKZ+?CI>!&06n()MqjW6~<o%k7hmKY#l*`?;HRWFL0JVV`)u$uzEK?S(vdpvKBaJP+kp_9Iv3oadwpi@ATwPUktmrAD&vxp;T#S&_Qb<AV8dE@o1GJI`y$eUIyqV=MI}>koVmnX7P}Ggx`QX-;rmQjXw#Mi=V9Hv0!WM<8qp&qHt&qTX39Ph>m(cgA_icAPUCOjwNbH-hfGoKyXN59eOTmCXtH`Vi+>|67{p@;uwwgJvZs`uv0U=kfn<Cvjf%jqaCuUJlRO4PMN{bBoW7Lw|Pg&s*pZ51!1&^M8&F<^1i#$6&{T2W-YX>Kf4d;LdS)uEDK`zUR3<akqF3K^D&aUR{sp=qw-1bB7l5e9qwf13Y(U_I1wto=!UvoMyJLTRjEzA^6jmIHx5zIye7sL;;+`5ghsz&NT=Q%Abwr7Uh7w3ieKqb0ve_BhbGd?3DO7k4<=vb1{Og&hxyYW;{<L*klIIF$*^6y_DwnYp!dJbg&=6DzDJ*AFQx{B+p$M$2r&~YU5mlU=f~Y7c6iA=ko>gOywNx9MyR4(w7OCA0N!Hnb(_!bK`?w*u6XlYCGp_hc(4{SwUMo&Swdl=5zkM',
'zsO;pCv~<8kBMlH^J)Td?|3fN>oM<n4pjutn|d-2=Vt^SWWYHkf!oV@Ue)!?-FcqX5}f}QIG+mZANX?`_D$eK5c?x=v_F47@Gw2ko34ZV2<+H(j^}5k#ra8r^*kppu=+06Kd`(s&Y=k`nth(<aJ|AgCV`oic;42uS-jqfkFnna;|k(@gTTmP*Li;Ru{k{F%ZBr-0{v>?9H&69$?zM2uBULmOQ6I5*WO!4N0~L>!cV)zLm;>ZcL?t8?jGD-2M992;O-i1a0Yj0aCdiihDm!T{mtxpc7H4D{`39Tckf#7dsq6OT7>RL&Z*k9tIj!9K6#EdB{8lE&HF)56B@0#D|N+g$vD*1KpzsS<d^p;-vWKj6H1MRo+cDqkNO!21<x;&x@8|=w+T6nsJD@jCF>fgZ&p_7dPO!xU5kY9p3sAYpwYmW1oKSTRf28>?8BS5IacbY?Unvs@ke2o-tT8npVj;6JjTcS_7ZT=`{EMp-23<<@ZWpyEcDQO^Q6>gyK)fqw7lp4z_@!)ub29A#}-?p-rE$^!}abNj(Swy9bHyQUATt8CGYxDe@pq2EK;X#nHm1fyWk7-$~*h|45>4>Q|ix6ngjgxj_dnQ>e1B(fA9{=4n6V?kiO^M-gn?fz1_B~k$T{<Po=(HBlrbxi>&Y~-bO!%NgceC@}4zjV*I_8rS7M<Tvi#k5})O{itI++6mR|!QhzU3S>Tj6t55Eq;gr-BPdy$u=1oxvby&PXiLe8&`N%q{-!~j~=t(S$`idUsC#f6$_gd(w=UXeO7x*z4`006lrU>M-VW@-Yd6)}z$~?CpVO~7frZtm#f~9~<o-^-pAJ6dxcn{AZsc+`l^9A*3JUbRi-SW*9P>0B~?n5D|TR0o_IXp{?12;VLpX8Cc=i@Q%o~cpL3(v0?fIFVp-oSg$$iTZ&A8|YE#nZ1o{EDZ?SLlnU^9<A<@wCqa{q?lGfO;>UCS9ZsV|~RgG;xL0ZLCrPbqqb_Zc07HlD%QCo}&5#Qpa&6?B0_r',
'TK=B(Dsa)0p$qDjcp}|Dqz>d<>Dv~R1N!1I9YsAJkG7H2q4d60`<;!%bGv^A0l(Z|)}Y>u`+YHtm;2>8_!0M$RwJcu`)8@Mesc^6sQXF?{ImO99Q?8SR1v(t`^XvCiF?1$OZUz<Qa5w!P}DPV|8BrMx#JdolKPlAFz@b#`=vhToa*37?rHac_wGsEq>lVp8~CyNmr3yZ?x88+_uTzAVE)~`qJgLGE@$PrIy95I@vYv1ce|Sog*~|&805WbEt2n7&Eb%`p9kdqO4UStHh0m7sBh;k(CfR<cQ5qEoo$wkW5&$#e5rS%{*F6ERqz;h(0!>3YVL)3bZZl(o_xYw)OT??bC{$~>0zn2`n7@7Eq(uHk<_dI1$B{KPeZ|XUH5;-__}VDyCC&X?*bQH=lg=cx=u@7Zr9OOuw&PO5>h{P*G>2x*S0?3FRqOy)Jt=%StoVyS5yG*yB0mcIJxGIl;6!rjru{Z$@`_w>-grdPuD2>IH}{h81*?_153*L^nNUL@VmvzbH`+rI<aleNuB)Wy@B(tM&ZzZSDk}jrEY9TsoPq~0-SV}*$w=66>kea>?&*q?{nqdi#W@bBL;Tq$`XP)c&>EE9!Z_rzOXk}SVr&`SAdkiaTzC|{*p^mT<YR_KI8sQd)!{B&)-7o_kIZ}D|K-%pze<I<plV7=aX`{kMo`jdg;8mA3WB1WeEI*^L#P*59g_P@I~j*{r98}??~XTb9ebjsn4r}zjbcB1pnh)I}i1zoGaT%UH>Ifcz)-+zu-TeGmn>%J^(Y|*POq$4VHSq`GBX+5kBaXbMP(j31`2J7&m86q3g~r&4Ay|4h7^nTU$^U%-Qs9IjKK<8usq2y$1U1tTrC_;;h(N`Vf?<4*%vXo)`M$EF31|mDdG*aprh+QR*FE1r9jV9}bZ^$eU!m!WYBeIfJH1y=3z!;Ga|1AAHu4*j4%ixY_{+9P!QV(x0G_{Qsx=cs|G5x-O~5Tn~QL@wh%F)^V?~)PKI&0&$V!N_)hoj`Q86',
'&h+Vlz*onySm|eQa7Iz7|GYx#O7GYX+;(g}C4COo-;;iOtG~f69m`F?PsgIHurJ4436C5z+DX5H-(peU&@o{R{IFxp+4+zozR16aL;x=x{mX-oI(qe%y3$>j0#_UzFMgN$)?UQZjuwSbC*9GwJN$~H-fGyhqvj*Te~zlD5H~m~G{gKkN-vc0ihgij>S<@d{5bM;0S|ZN+%EOCv%1uN3)O*pjx>v;K6b<h@JvT&S*fobFb{a^Fn*HzYO6^d^Mp0P54%f&-?hhg$|3c?&%&<kpYwvZ+uzNHUf5qIT$g?gz2xr?9=4HmM(w3iSNyotEx%k0_+me|4exJ1odft_Kei6`V?UHFuk>|T|5ECe=R$mK-};BtA>XLhA73l=`0T41Nj>wWFY*5Ng+t{3=USv*`iwQ0Py3WIC8b`v)V;Nj9rjrIL8O-F9d=Z%GpIB0-QF*-k@SZ+2t2TN?S{IC_LxZU0(;v_6Qy4JIK+YWrlnF!--vkWH&JId?807S&`7EG9xdN5?}0wrOP`VZ6`v)4FVa@tyFk9NQa|1&*NwW1_}!jmJM6-qVG4MnJxw>@t3726=&3y{5BP>XNa{w~%`SNk{Tu0zq1=}K5Z*tfzo7G=^aY6D0X?<-*aZG#`?4N>(Dq^7ZRty~;fmDD-vWPYd$L>l3p@}WZM!4&=xsM1NneU9Uty27i@I4-k3T)$-*&2`^piN&Li$x48UbEl+qY8sAnf|{n)I*u4ji;?Ne>*e{a!DX^syKx{VrDRKPUYHen3xci}T5S=l4D!{VaB%Pm68(-|%a;DP<5p+a~@d{RPI|ohy9`at2D@f>8^kkHKxJ|34`ILn*U39r-X@uP@RUqkBX6X<O$5(nq6XmRRYZAoVD1Eq#d3Y|RGA?;3q9CH*WqB}#vW=h8Q#Mn^^ZIlPws1QmO3kv<T1=>t(}nT&JsHqzIjXer>Jtxz=V$d<ppjC-C*^8Yy>Nk4<Adh%XbpUHe?o+bSkGPaZXNncy~S)^?ad$XmUAkUxb',
's@ykaN%$FCiqqidw(vgSBet;OsIzVh%_99FLh{MD2Dbpev;|50V+#sHT{~Og>fs{mKM<Koo+~g+`Yi-TZ4lW;{r>K!^80+!*CFUaYq|bp(W8*=h2{Req`yYkoc{8B4@FNR1g~r<@5s1DPKG_$(zKZ`Wiy*fACU|LkPowE+9~sqHA3b&`-<5zjy225xJKG!oCEI4xM!0-8AUtB$h@4A{t~6D&yl_zKh%DaCydgcV>fWkR_l_yM}3pnTjQS6&!YKv(StV2MUOl7IVk-$I$aliixoZUe|)v*Wp)|&U;dEi9M=(k);2jB{L40@u=L}Y*GT$rELkFYyjmxHIW{bm`);cv^RXwF^#3@V`<vKhBlrc|g;j;bjzgtS$^A{>CAR0?fOod{wZwkDH`*lqF~&;Y7T@i3f;V+AuJ+JR(*GvScIhjTbrE>JJ?{qT7g6lN0l~q-(to1HAJXTiQDf<&(I#zU!CRBa{%NES&&Y;a!SAiJrLRw3@i$9uNI#Bs%fQF&JErKwk1T_J*#Eo%9JAlZ3p}zv-V!7IEb0}KGN~!V&m<<c6n_{9{dGi^7aovpzWDD#0mz3r%5Fql>ZsLK`Y1K8A;0TfXOr|r8njG!!BOClV|q^Mm$YOb@XhgiSLsW!w}klX(|LirjvEz)M?4#j`s0o-*T91vo*L5E$?_67<xIUocx=vL!Uu~FDk?l~=3?P_7d{FP%;pxJxN4#B$dZ#)o+;m3dr_{p`=s=xIDL1R^ouY{ABxwl;ZK}@Uj)x|>6?Q`xKb*}U$~+!NE}dn8`fJ~HMZluT`ez3e}`Ur=>svUmGqOCaS8m+wW>vL>F?k(N<Xes5|`Xw1>Wd-yGZ(QIW{8C>9*X09=X#;UX*xjDD2o>@eA^E?k1xp&gzz3zBl46;t}`sW8jJIRogI5?p=GXO5d@Ya@_|;#3}C24Z-W&iMyss{G1Da_eAZH__<V5>7P)~3S9PdyaT=S4A~=jh$$OjKb{p^q@TggGvH&MKfgDXyhZ+@(r;}H',
';#RNbIqcG#stM-9o96*|hPOiBZ_<|_0^{QCajBw|yOq9YUbUZucVlhnwfAT_$?M##fcoR!clD(|oTm@`Q$pxs$&W={J1u?ZQh;|QH0&(-$F3)&PX4HzpQRk`az*lpsilwIku|{egquYrKltwY7|DMQxi0zAY?3d|`S6w0y<Y-9k=Uez)SK^_U-HmnbyDYk!E>ntzvUEoOybGSQV;*$VyQd-Wt!Aa)lQJSf9hDNyIdev`u<g&ApdSXRr2}+=1bkz$#GIwb;S-@zuJ9V*3mB9m?G;^uVo!4-i~u7N`M9Dos=x;agIkRQE&ugy-MjJJ2%2PA7x|*oZC=l_ZbMeVOTfFBV)!u-W=Z$^8I-EcfuH)f6=5Eit{#_+&ysaL{p)86lAmVogsT?-UB(#i}OL6MYqh5+v0G3NOQV7&ShyH<j4s5^$E_0Xm!(ZK1Q3S5YE483tX85S+zaR(P>+~&j#5aw^y_iAK;vzc3DL&<R2^gK>qm!=RLHK3YUlc-s8)^Qm3Dfb91^hyR47}PU4)2uF7dSx6^XpVaUGAYCultk8@GF#nC4rw|>TW=}yiIhrF8;<E;C*4d-KYiJ39J`tS)AAamT3^F?JsaehVLpe)YI=sP!)^GL&6%lW10bumx+)p_y$`rS61pVFV(nicX<U7Q2cf4z@$eFjZ^oX0Xm$azXbPM7?>Oo1Md_3O&HtC%J@CutZ`c^BkwX)i-AeT;K9hHbxAgglW#4|#K18pzil<ouOAKfY%)x5RagX?x-vnlWEYC&=>UM?*GLFfPWJO$#9hN8=ouapLbd|7l$4!8sh`#%P@9G9GLUJu_Zx(gN~Pi4BmSd^k^M^lZlYB2!@TN|5O{EP%}Ckn?Zl@=SxQR{`gpOznze-c5ajq1UF-=P@6q88N*fmp?H=ZY_avHys`%-@7>TD&&Ky7$?*F&N$~~vS-Bg%=&{kUuaHI7{51XnSyg#=7J}1zSCUp&Kk%%7jPcV+-g}q$nLdq-pxEj&bgW=RIx+Oo{9H1',
'uh`oXa`Qo)yE5-zRuJ+`OPupEUytty`J@ldf0{p>dI}l;H^$MT(cpKMz%LkIOX}S?mu1P`4Ck*b1@GY8nWaoY+}~2YEA-mZXgJR4S=x5RIY3MIf;fj|8F&YJXBpiTIAEFlM<V2$cXDoXnN!YNu74|X$9CAa<zPdcTeY0JhV!tN%Q<nb)N-da&WT!{_QZTy-Zh3ES-z*ld|4cOVXsytAI|+*&Er}_hVH<)SX1wZJy<g@#kodnt`>4Wwcz&-kR@X9zSi<PU_aJs_h66Ky7%RrY?Hk>ziVyXORnG13%j&-?|uui?+y#(;A=Q1XdQVe0&?tX8|0*h7zgXL$MYfQRKmG4>%#HxA(yShcv)A^$2mmn`Yxp*H-`d`tlJmKeRny5AJ+ZRz)kDndix-cSA@M-Plw?AtM%+5ob$6@EPoF2%4*D`_4*U&rS(=k&TU)oe!zKT>w^P-LO$*+&-wH-&N*6NG>~z9H5+zpeX|vIVtuzB`fB|!@&)84sHtN8d|2uwe9Zy<uzu^0^Muy#3ui+9SR&_r{~8IrvHmO}@A30?EaczyvWZ;3U*w}UBEOXs`J%tbT{0hkcbqD}|0du6+2)a4e^Xky-|aZL|1%l)Zx=Jj^Ud!C`K8=idH(Z1AwL#@pRm3kGXe7LX8059>wR+G`Q=)iKeRp{*i^<T3-HwXXtzAqgQ$KouD{fgao$x*#{E)vnTK;GnV0qQ|L3{_m#t?ka^Dk+nnNB<FXyli#>u$u%Pe}ZYo*M`j#TF%x6B0YST`!7N9zVw6+L<;&%dIkoa<il$0pIY6u?vK+#&M2nYUz~r{xnpo;(paWu5RKUi3GQyw|7^Vn4(GtO7YCP{wsYBjAv=&jLCB-Q)HW$gUxx2QjrokK2ufJzHDt#(7z5v)4G6ZEYMXdQ!g(@XcDg6YSYq{Wsu<welvX*yZ`XkfmS$23gz-JhB#*bKTa0QF7nB(K0_dtH`)#tB3Qz)=bT0oYS?&d3kGOTi}>AMH}!DYe);=t~H>M%%`aa',
'&goipWnga>Up`#d;z^fJ@WzOG1eTxQ<-GZ~yYMHLPX}KJJ}rg)SzZmV2KlTpc#P#yo=cGT6r5+Z+`J9FvRqwv7xLmj_&>|pvLgit_3$T_qnG6G2WR5_Eqm+XT(f0ofIP?6b2xWz**I3NyDnP3zv{zl!P}(=AQ#oZd3?*<zriysGgrtt_^DOoeiPrrUM=IMMhSlBg5S0b{}Z@o8QdOz!_v==Iu4dz)1_WZw~X*tmY5xKopxox11v4i%l|iR2z|6PxQ}!2mO5?3&s2XR?_IfV9>{VJVQ-dFO@M2bVwb_YEQKn`_~+dRJF?`=0sn5vx=@}sqgMP)nm*unmWVs@_t4U?CriLe;DW{I!#Qh<wtZ&!zXZW6vn%BY@rQlo-2V47z*F<*NbpGWyPoh1=9h=SBg{{YIQMLR&=~&Te0#b0>uayT$IO><LjTNXdx004Pi}$VGavbg=QkgSf?k_<ca-~XUxN8FZ@La#F|XI*{mrY3he0mu2Hs&_xIp~(oKw(K^YqWq2lH<!;)NHK79KICz5IK`c+^`k4_*siXYO|j=cUa(pF&T~U7W%LI)=F*+vEn$nwyuGx(1CJ!%vv&c1E26bB)2!M|0)zZ-uYS0xvX|Tmri_7hMB;G#A)}`Wfcj+rbmf*>_=n%$asOAk*#v4w_T$f?b$HcYxoS12)Te8Q0=GxmmLeJm2J<BXtZM69+^794>VZzV-lbG<|3RKWlnj4R&FARs?>-^e_YT*L24Sy)a$-R#$l19n}3W{doxSgXzSIDZ=Aofn%n9?dL)6EZ13hURunbX@ebwDok-Vr5?lbEx<9;q6rvR)7+MggeT?~p81;s{WguiEqrkFI`|pW@WH54Vj5KGtjaTigQo6}z)MV>)?X1G+V>k|%V^-Usj*jhXuVU=TT{)+@Rz14wO~J{@&P)?l9y3e#Z+{<JZFJgIOlK5ZIrqZ*-p!NWgG)MFr_IW^&?XJ01q$)ZvtMMEFEOL^bz2(#>9(=JB+R|uq$JHA-uQo+e_ey',
'@#8!^zwu3F)FCiF`+@ryAFe{(5aXRXcrW8M$7A8^>%fDJf7Zje8IRjhkHL6obrZ<FH6&ix@m21>c>(ajxW1Isky!N<af)&2c+8J+L3YF=##!gVzl>9RNE|THG7d6!^Jd7AwX#cG@gDeM>^B*F&e$^>{F<@zN$88QeOu_ev87Yuk|y(kXU6&kfFH)17vy(UVxUjPa&FkOvE)41m$7Ib;Dj;%Y3QplSE~Y$S--<Bj2R}$bEi%RK4J{t2|sNNss_JcG(Ch}8nylKK86Gh^vU2@C~?-$T;REeFGqmahWGVQKgRI#IrzBY$w1({;l6I3#A6GACx$CIz>5v%4ggmSr)t6v8jd^!k2UP?iTVkKoo?8lVe2&Dp5gbjlOf}_V4MugOM|}|7G9F?&uI-FW0>|y>UB)|1#zokoE76__+=?@)i5--JYWAqz*|GFTF^^Fmxs`6Lx*1A`G!_*;DDj&G}wuuK|1gPL#=JlJ44m7sOMlPe-(8m45d0?d<{i^$a58#030ynj*$1u_B-&>kg=4!SL(|$o+;W(osgiPuy=!bGW@(jmlk@cPuPKR)H|zyPwRg^RO9D<z)$@L1L|7nUo8g?>Yo;Q0(t)u@*MhGoiJYdD<0sB{`>;@`{@FhNBz+Y;AQ#)o#7|-yL`Yw{kG-6SN+BkQcq;fUErU7#bEFa{o-(`Q!;lK-d{hXF`ipL`ETI1e*6OX75(TE;Pv`pkH9PS17pDp^nJ3yF7@3mg8%4a21p&0wvmX3^vzGgZ|EEKg#PI3roedVtDivJps&;$^`7))Q)8a=#m`~>^@WFSh0GTvd59eM;dk{}rb5s4>B}N6)u;LkbwTuDzhmC@f$d<QdQ-|dkeX}Y=Q{5c)Thxos!E-cpBmt)?#pq+pSt(4QupvxdDxHcsRACWyMG3Due<dduCKdV56`8$5CNRfoqh`Z)*ai3_s|_21UuI4sQ`V^?Fd4=qucZpc^}=n9g^2sH6FODTiOiqqi#W7sk<^;i#SF%^|9Q4;y(Cc',
'-Pn217u_%YG48sd4PZ~Y0fkU+MAth7c%rVWL!PVSGvqaNZO$Qop=<sJ>_gXRIsBcj?iAppuEucWPjr=gz_04cN&KTL*%0xbu4s+gl9#FgJJIDX4SmsNFOGT9Wr~*fOdBorB2$)xztDx2h5yh62#?ko>%jl$G|i!(T5o5Jo7OQ1xTgL2EA(3XbrJlU_QO`>$F#4{VEnYto?@J|4_(k-?cGSJ8*`&1@)z36ZGivUbK@}n+Ebf=^V*}gFb>)SiNHDSp8OaO?T+@)Z|&x};APtN=ixuKaSG;9ySxnYAKFDDkWbLgJ&62*c7_LeIPH{*h|{#cPJz8>V{ZW$w7+DM{NT|3@Tb}Vr{Hh3eNw?*wcYz+e6*b|Ku@*pqn=A1aT4%B+w=p*UE8p}JV%}VuzPKdEQrsvmFLL&l-GbaYD*23e;4~K`N~3F!Lzk_Uja|GIXg-IG3#6K32nySnI)eYFXNDM9PB_F79!6ZxCZ#8H5WfA`OoXnD~+!g{Jh3vK|Vuc--i60=4Yc$k}q|`4{JVd06)>ZZH95tyfk54HBXMkNq%)0>`-&3DENWq`rp6}&E*5&qnh)vz-`T$n$QEyiB!NH&EfaJd(Hl%z(vjOg_7^xF%bHs+0qDcqUQHv;O&~V>5wnbtTF=sG)v<V=W7<eLmaP}`vm%+nRyrEubFyVo_Eq6@NmuehZqmdn3t%pr5X7RxTG1Hh&+^LPzvP3H2v~|w`h9Rka|MhddTlO&49jY+V4eNscHQJ^Q&nQhWwqTNj2!Hra`Raq3aw*oUN(hf#1_qtqA+jRGf-Dr>5*Z=&7buF2v88=veSsP2mSwAPW=+PHFNi1&(QQDzGa}w%*|3noN%oH)zr~SPGf?0_;+gvYOOY3Ogs`8eA86ud&{jdP$~k@_U^dcCJz8Ngbnvf}<c^H^CPb`*6hH%HLVx$CU3kF|W#(N$@Mm$MWDI%3Ck&R(W+o^7_vw1DBL1O{DJ9!|bU4q}+AG?<=?N!XGQw_raejSC)eJDHnf*',
'e^$<^^_I?blX^=hJAlV1$J)U@l*1h{KFWa}z!zojQ1Dn~*A&=|vST&!0?O8-@E6LaXImhDH$tD4bw%Xgaa~aVP+7Sg_^T{?0DVxFq=mgJ3uELx=j{LvDswcLS7l~PdEaUK5w9pyQX_s>CXGivOquXa#${Y5_yuLmW9Ypys<n*6h-WJyhxI_7QyJ_;T%-(~iTF?HpBsKf>2nGBAf?wp<h7I@8Gz$T*Bdg<ou_~&DKRyme@ce{;DXZjF6xshtvAR#v>c82T4~-A_2`tQrR6;vXO-(U41>NX^$oI)Q&*AktgQjgDK*WoKc#v~@K>d3Zt!fSauwv2l!~3@dCE^kJg$`854~4Pe*|ApN@eRH>qDK}LPl?fKU0d@!FQD+Qr}T2v|jYCpkDMae{UJLybs|=l|1!eH%hJxQr9bIE#QX|bxX!4dyIT9t6k<Z%Y6BM=Hjq3CF2XxvkY_PIn&odJguaYx^qgJYjT~`%jEeY`^oxMs#-G7Df7T@C=r4aN(!BPKg=z99O?w_Dj^A_MOtJWgEPo;1(g7fDuJyvvVQh!Yf0qom492ls3-99f}bkpL9&i!I$2GmI{wB%;1B;0_{k!xh(75LR1{eRc^E~vPwYomLEck)^NPscGN0NY@cMrMSoA{Ey{O1c@|?<x@gg_96FD-c$VT-=7M6Kc(!{6_F7x{jB4S(egfwN6k5eXpl0V6x<WKS^`IG!f{v>~rKgpluPx2@Ell)2k{QvnFBtNI*ko=9(eVoV(vfo-$yOYSz*cYW}w_;sV(M|6Nsh_%2F7vzOk&G`TA8Ia*^)kh>3;AUwpc?jzD1mm#7Y3h|JYDD?Sf^IP_e&l!;vw?oN@QxR3n^&^V_if^_d)WI8AnP!ElYN3?2+x0TtCMp$y4S26YGaczWegLh4k2`p%kr+^#i5&ChQYbN@vEpnNsc#tV1Z3TF7;(rINf~&9B(EpwxXLdANozu|A+QO_29(RS5fOly)N}KNa&p^2^=o%6s&>FYniXnB<{{<iUD@',
'G9n)Plaw(xC4V^KjJ(%x7qMQd%y=z%xOt&+|HX}DK2~gyeD&G@8OMziv9Cng9wNWry<PHd2ir=%@^}WU6DeoD%RF7Yh4o40`Z>Ap-OG|Ue*6OaK$KTT^j}aul#%)PHcs-Q@pmPU>M14bBARVjFVdLvV}G6|_&D;+nv`u{K&H1~T~(9yPvo67xo5~aPN7a%pVgG8hV?N``Do;;HPwp9I#S&-vR}VRBjf=!tq03>W7c3CH9a2TIWz+@As?<8-e0~SdkN#DnOqq8Y0b<v*e9=9m>&CxG%MC(y+E_RF!HFHt(Rn<|DL|`JV(;Vbx&WzJ~qvz8CXx#+-f5IARcAII=$wl2YE})$H$o=|2iY<Gp^mT4yf4-yU<!T$vRos4p~P_b6D2%vRuP^Xmh`p^~)k=>=V?ME+Ff1m0M%|NLza*^1<51e?nihZM3qE*SRXzwY0sbpM)HIAM3N)(Yd4#$HZ9Vm$ftQOP`H}(O4hRu3Q8=(Qfcy{YJZ^3-;M)_us&N1MP{5uovz5-N@f-Z{$S&Rr_Ey){(R?(?GAZ9~WZ1NBc7Z>v3B50_+pe=~6*obOFntkGhCVSU1t7|6SIlqw-_@M3;Xb=1Et)JoeS-%3pw<>uNNX_2dRmWPQ11cj&pU<6qbxr0Y3K*53!|u^yrOWhUlJ_p7Yq>!z>3yy+HX#rmdh#YT*$ZbM$IL+Z9~gI?<P=SO~AcYLd?ho8@hbu8WWH8}U6dyoctqI)q5d3@bRjhwsqYXJJI=v+^+|3$B@A^Re%+o50j6k)ioKHUJ>H<9fMey`7$9pj-dHUjUbFM9#&Yx-&tSa;CZYZC|AY!z@v-|mU*+vt`O<Eigg8~eod!$!)!iP&{m-_uV%FZ)7fy~8?>ezAgmdHU56a-L&j7VM|f|B)MZs6UWb_IaGhf&D}J^XZUx*I&0_y-t7cFRW+lpWTE#>)&ma{S)6}W#5Rsw(R>z490q^!EhPtWrm<}7=J^`0$49Mq`!@Q0*34Zu|8tRXM)}tiY}D>',
'I%U!VcMX*n<6MHFcA)IPX*2@3ZD{!zcx~tyE&X%4{|0+A^t%rHGz<%d{Tjwp#rl+C;sC5m8m7<3J}Se!?Z8#T(xcc{Ylu57`-L_h2d)`*?1UZ}_AQqEK}UxIzYS+<%6nZ3!~Si<jq4R5?@s`Z7@p;o>%6&${RM{4ZAJd|M*0UiV}Nsp#9P?sV>FbKzC!^E;7^TVuQ3kB)H$Ia#*7`X&S%U%1NdglvkiK0EOZL{1dPQm;W>?E&!PW|vGM`zZ!y-C{W`|_gXCOG)AH~;##RZy6=R1#un*GMwJH3QvDa7WGdW<S?57**#QPdY^}_xT<GAZscQa1TiFH}y^nt)r<D4Dx9t)qrZj8${a(*T*73R^nAp`ufaZ4od!1#v__HNws3if6^xLfwK9h3bB#?!g6j%Pe~1M9%X%RS^i*MG|XxI6u2pWcHfSm!f7En7us#7x+U@$Gfknemej`fmJ|8|zZWpB3eO?6M!g=&l6)H74c*u9&oD_!*PoF7~6DEc0dmWKd<<cNqE#e%%x?7<OigjF<hA>AK6l&5W01pJvvKa-JuuGjPL{YYFVel=nFH7nlm%k$sMZA4C65#cs<vo)Sl7|7K~~*Jmo%7V~AQm=g13s(c*lpQdUpz;{eFUIJfCwcE)4(Yj}W7p4Z8WZz_?UUDv|$$HGEso6E`FEF+EDEnAj{SE(YYV#F%V`_IF_Gap^OZsocjDlU7I!6QlO<kVL^K}~me`4yP0pBzA`~~A|>h&6a($uFc{FJHhM7e&yqcU#&U(0w6a0AaxgES9BI&n_QH24|j$uwjy^w~6Yr0kO&R$!3GTk3w*?go*cz)wsg+C39_Qr=@^_*aol&xxFzTjWMLA2m|?-2K0jf9Lo3JM}ueF7Lzp@;Ufid`><$--GYP_vCwX95^l<CypD(k>kp7=D2enI4_(h&Ku{E^U8VVyi*UT7t|B#4fTk6MLnb5Q4gt?)Kls$^_Y51J*VE&4rmv&6WR^!h;~IgqutRCX_vH9+AZx^wQJhB',
'-|mS6#0BC6af3KQTp`X7cZfs8CE^rui#SGHBhC@`h=asM;v{jCI7(b4&JuTt!^CCcG;y0aPFyF>6Zh!{=ojcG=r`y`=vU}x=y&Lc=$GiH=(p&{=+{&~N5ALygY=8^lk}VPqx7rvv-G?4!}QDa)AZZ)<MiwF^Yr`V0ptbb3FHmr5#$x*8RQ-0A><|GDda8WG2}JmIpjU$LF7d$Pa<#f^C<Ev@+|T$@-Xr;@-*@`@;LH3@;vfB@<8%J@<j4R@<{SZ@=Wqh@=)?p@>KFx@>ud(@?7#>@?i2}l_!%o`*}2ZHF-99H+eXDIe9vHJ9#{LJ$XKPKjQ$#1&k9IH!zN1T){YlaR=iN#wCnX7`HHvVO+yFhjEV@2Qe=4$4QKv7)LR#Vw}aei*Xp^GRA3)+Ze|&u4A0XxQ}rl<3h%Xj2jt8GOlEt$+(koDC1Jbsf=40$1<*EoU6vYjD!7gG2>*$&5WZNS2NCL+|4+gaXI63#_f#b8P_w;XWY*`0P_OO6EJVUJOc9y%rh|Wz&r%=63kODZ^1kU^BT-^Fz>-U2=gLpo`iW5e;$Q-73Nu(cVQlec^T$un73gbhj|_5d6@TM9*B7%=82d$VjhWkCFYr!cVZrjc`4?pn73jci+L^PxtRB29*lW0=E;~hV;+rpHRjovcVixoc{%3kn73mdk9j@j`Iz@(9*}uK<_VcMWFC=uMdlfqcVr%tc}eCenYUyflX*?%Ihprl9+Y`e=1G}1WgeAzRezq9c~|CPlRwFy<WKS^`IG!f{v>~rKgpluPx2@Ell)2kB!7}W|I-iJMrg0dc~2!+&hIKQSSL_4W;y4p)yR2MT^-avQ}pj7&t|+K=YcI|DdZV2OU@?-$H+O&u(6T{iuj1S4NB@2l1ECvbU9?^XUOv_QN71O<|!}dSqrw1^N_{%A&;+=Zh>_arD8eB=T+|_d8xWrQRhZ!GF+akb&Q;wjhTk^9Hqw(tXC-m7RtHG5kn=vIBs5J$SEJ?JoMaA$U7^`s-dozvaTl1',
'w=3IzmHgs98}hTtiIs8=_2Msb-uL!&$xA)G5eNC9rkr2>`BTopCR~#|s`(MVr-?|_5i*n1W7Xu9bI_XNFH!G9Q~8e6Tdf~2=jz*(m-GAG_u|~MW{A`Q)r?Pv{GMi3fSf~Lo=yILQw-)yv+tao^FJf?4>dP@nIWIul=JXk&SPCm<9?2H2dyOy##tMw)<uZ=fxM))*mS9jP^B8yO|*?ON*#%q2&vmJAg9Q&%_YA+YmL;ASY<*!QoH>(skd+}v(&G+atr5vwNF-Jok9C$Y%56b2)u_b@HeUBk$xN2V|4l6VVrg4ilKgvuHHh-v#z}rc|~2nxl&hTY#uzXZuT8n&xo5X^)z<1lDe*^iz3gcyPXc>se2QSdC=KXK@arCoERT{WL4z%^||^;ot9E@QWvH6Gvs;oZFAy0ufFdH)LYZX-om`==M<AVFl$y}olU<d1bU$Vb293r=<n%K-$4I<vRuy<g#5n2x)kqkNS{mA_wq|U14D%dvYyuP9e!u%6pK8&VMrwG$uLRk8yXhZM%^33rq{BLcX*VnJ6;LHy0GEt2CVZNz8A-M8<jutJ7ai#jE6DX9o*L#-3)qftbP;vU~E}a)<t_Bl=`ltvS7W#IAfw*XXSH@gYl2TQkUxFDEa-(qq2VbDqhw}<Fm;6wV|r)CrH%}^^{CGd%`YFC3|Ci##E~V>SLH%*OYo)y|bYXiD}ditRt9a?8N$^X=RMm3EQE=e4CCh!n%y<dK#S9Ha(vyb+^8MmwIPPC7j1LhfctFm^1IfdZ4+`Rp_<3qO1p*8(x=suO0Vbz12M6SE++Gt^(>|m}kF-KAKnmf^{<U4u{lXJKhoFV7|Ir>byPqhJ3&IOA4v4=E)*;T&=04E?{a0)=4Zmk4v4s;(dV+mTEfazop4I;G`wyHS+3~0r@dsme{sZ7j}lE%Pq_LNWH&Jb)ZL<{UK6c@Qlot<z`i>+xKjr)cyNn#(IXuRRQ}ltj3m7w=zXDtRGr47X@xx^ZkuHy|vUrsc%|64eE7S',
'8;_AXknPXNzk7X@I*!A>V_niZ;V$lPowWk_dh4>vSO>FiI4|GZl~=BRw4ccHvoY@0TeGo_Y<=1ncxnBRTk5R-JW~-ep)|%jz%&bZ84z{?`ziv`UB!F^L>)kVgMfmu(BptoS%A|4mE-W70d?H4|A1zNux=R8zB2SWpnCz-mk8)@$G(Ap5liHI<5OaO0;cxGI%>e&Rj}89rCVT+0c+;qy#h8j!1xF3{DAe?fP<}}ZviKM$MXlAy9_%HxOx%e6L2RE{x#rn1L$4A%llYY3iwbG^AYfEApBE6`~=v0fV(%=ivpE=SSJfKoW;6NU_dnLY6OOkK|QFzlq*p`ATaF$tak=x>V$Q&!0cw|TVSq9sEZJo?*Z0%0t@S~ZWkD>hrI=sd<uLHEIS=I6<9G0?-N+1FP=NFdK}in18ZRzl)$=EFz$g3%3;0&8(oy=X_^;b2yEUN_7>Q35cDpvb#tuK2eu7`od&jxdjZ+O1pge^u?prnFs2#S4FfxsD<RU2x?6#rXRQ$F1pWthDbZYH-F`{_&HwK_Yo+@8#3a|_^*dInq~1SmlIP*`@p<|Dd>_6a-<R*t@!<GyysGq_p^j&4lJVtubNo3UoFC2?=a2Kr`Q?0vj=87m!{#LVq3R3u$FEP+FX|ihkNQacq`p#rsn67J>O1wH_CWifz0iJWPqZ)E8|{zwNc*I{(tc^rv~Suw?Vos{;sf!*j~~Pn;tTPH_(MD*J`t~oU&J%w8}W|#M?54x5-*9L#8cuc@m9rO;_*NDyh_DueG>d8o)h1R_r!nt1NsO03;GZG6Z#kW8~PvmBh^3AU-|tP{n^#s<y3!XPU8RQ59uH2FX=z&Pw8LjZ|Q&OkLjOPe@*}G_viHQ^!N1t<OAdf<O}2v<P+o<C3~Ay{_!&jA5r-U`HG*vkk63cknfQHkPnd`kuQ-ykx!9dk#CWIseFw5%+J@z-^l04@5uMa|Huc)56KtFAIT@lFIB!t{^{qV<fr7T<getj<hSIz<iF&@<j3U8<j>^O',
'D!(S*_VaJ@aq@HW^|;x^RX)Eq3BM=bC;w+W!1#di0^<iYo?v|8k2e^9Fdkui!gz)83*#BaH;i`}|1ch6e8hN(@e|`IHNN^M-nym6Uz62%?2sCtWl9pSF@9q_$M}x%9^=0uO+wZ9Fm;l6k?|wrNye9qHyM929#!L0#;gALmGLa&TgJPLe;E%mK4!ek_?huE<7>v-jK3L=Gd^d$&iI}2JmY)D`;7mY4`6<P`2yw-m``ASf%yjJADE9|euDW5<}aAfV19%74(30Y4`F_U`4Z+&lwj%i;&bj1c~viR82Z=v664JxYe+-PgaPQY<ns<dUo4-edR}P|`gbb1%y9Ij@!4;qAC%8#LLVvL-zm`7%lE@6ePzDwM_(M@m--taKb}N?D&PA+^bPa9DS|#kzL&+yN#B_;^#Aicx`;kuzWeRbcguJCy7ZsFksf{Zd{=6$lYTV~(5J}vXL0%e6K?dW@*P<z{p$~8{3`uz#>#VSKZ!nVzD+OC=hL_DCHjT>R-HzlHQ&-nG7bxJp+B5&)^>TmsS)UF<omTf`ULsLOq2H<z7+k0d;_PWpQ*2RC-nF6bxkkh(f%;{v-w(<L*GqbqYd(2wST^r{yb6W$LcFr9{uTj#mmcl6wHQxSH7G-(09p~c|H1Q`O;KGUoBtQ8T8NdSql!8{y)E>&rzcLr1WS1`5yhp5<l6WL%#ZfzNLwe?noc{TWit3GVxMt>DPWL5q;Yd56wc~#>8D_=?lN9Kl)T9u0Ac-U#vynuEbdd<olCqN`L#&4bbN)ad6eK(!V~Z^r`RYL?5NZ76%VQ*6+Sv`qCSupM9D6(zm`y82XPT=8D}Y{pO#be^Fw}yy)YW7|<Ji#S%3uGDsi!<G4=3*IS~luOCVu_J_BmfBW@g7~h1mD>_MErhd?)gk443NWZ2p(*J$MQuNzOm{%J8wi153fq70C(*}B$FzB)Ld+*T*{g@Kk9mO~%G|4bY`m+y_e(vQ?qt9SMQL%%BT$QD7dxoI{rQiDMGmxfJ7+0_N',
'3Hq{m|N16<#oxJxO8@i(>05QfjXu8Kvp=Md_~93@JMSMCH%mYC?dS{XT{>O*Sk3Azeae3=0Xy)HFyOx4KG)MpU-Jd9S8wxX(g&+fxZJ10WxR*C*!aqjxr-i?{^YNrAKviU7+|ld6#9pGJTK9A%Jbtl^y~Ay%^N2Dx~>+L{^ET<Nni0G%)jTrHt7qttrqm$6ZandGd&9@%J-*b$u0fCk5`kv;BBOjcXt==>uI|Led;`oN=iTTY7fyT%Ts!^+`nLE^sDk@J15VTy5BwN<DCZm@jRMy($CC3Oe1~Ea!KFr=g*o$-dPMia9?bK{%r2!5$G@J-u(c2;ocaBzFqDWeWf4v-0~QI_vDB<(pUSt^vmvly^i$J-aA41XRk(|Vt4)7cy4!<Nw61p$qCS7cm9bxAhXVdzPlq=0{`5Bdtm2o%?;S4%WjiCZ(p)Vzv-7v(Ra;tZ?^PtzI^M3^n=R)yLas$D1G3zUfC~ws|!iL=|wB0pY)6rr=&miJoHa=4Noh5rTc6{pFdZp8o(h}i!aavSH0zU9#@s7=!fSj6)FE;;H^AQ_G8jVJ8fKd$k16iq>uDu+}G)yhWk2yE|WgaANHcZne*9WeBXID2z?Wsm+M@SzR-)LFY<ve=u_t0(E$DKo$HU|eVxloqYtoi?iuMDJ*AcOZI1QA-kd|Xp<ko3PcQV#b9Rm%EB%~<(SOp}FkbpU*Z3&UQ~oph^Esovurp`=bkd(XdriE*Gu^}^(x3S{aL8%RCH(_6b9+iZW;680@nbgnNjpA7;dhSbm!<#V{c+Mi_<A+1^jl6N{eq7tNI&BJKYL4mWe4!ju`%F@^i?j5e%Fr0-7sH{+1sVh@)RBP!ZEJ*3F)8w2K}`i1N%iv-()k!+tF$7JL#7^2>nzYO)Gk&PjVLEg`;|k?9v}Om0YJ}KIxxWxC!vwk!ObVY0UOO`X^_I#<)6CZ8#!*kaNoQEJuOU4qfLT(*HOE^wDnrg#NPj9~UvN_D@IgKK3`K&{y03>>>2Y{!s4~',
'8c_vyV!yf&c4j~C-Y<QQ2Lr$DM}5+__rPk*k9~Jz=&yZynupTgIR1?EH+~I2YhU@Uy!5&BV!rGPazan+v%8^Rq<z{U@CW<k%;>vlAHM?r-aaOu+-JlM^ntVwnJRrQ2egpBnte*a|J!>Ml77rx%1A%Xj%^l8f8#kbrN8kb`ChY95z?n~8^+mQKN|CGuYEhK^zoboeYRJv2S07Elt=nDm(L)5J<DX5KF=j90pIP>gYDAy_z3jKUMMy2#-4xLXXzIjj((Z;Tsx&tXO7P3-)+yHSDr7c1^RB!l+X?`qY?dZ?dhZ7-|cBz!@t>6uLobUM;a{BFZmbr)3rw!G2iy^P0`Xnxx)<UUz$&T9}*((8EjrJeU&4nuXJEpng4)c@^|Zbc`j>y^e41iHs2RnHjVT}eIxI0UOPnOFu9Mp<C3J3f9Lnq-|;%UF7Lzp@;Ufid`><$--GYP_vCwX95^l<CypD(k>kp7=D2enI4_(h&Ku{E^U8VVyi*UT7t|B#4fTk6MLnb5Q4gt?)Kls$^_Y51J*VE&4rmv&6WR^!h;~IgqutRCX_vH9+AZyvc1=5{-4h3h3&aWH262SALYyJ)5Qm6M#3|yIietnzKh6>Nh=asM;v{jCI7(b4&JuTt!^CCcG;y0aPFyF>6Zh!{=ojcG=r`y`=vU}x=y&Lc=$GiH=(p&{=+{&~N5ALygY=8^lk}VPqx7rvv-G?4!}QDa)AZZ)<MiwF^Yr`V0ptbb3FHmr5#$x*8RQ-0A><`0Pa$vd^BD3P@*MIW@*wge@+9&m@+k5u@+|T$@-Xr;@-*@`@;LH3@;vfB@<8%Jl_!!n`gtUIC3z-!CwVA&DS0Y+D|sw=EqN|^FL^L|F?lk1GkG+5HF-99H+i_q%gNLIyq!Fryq-Ltyq|FZ;{wJBj2jq7Fs@*n!MKBQ2;&mQDU4ef$1tv8oWr<Bje{5$`Qs$UO^l-$S250F+{HMIaT((@#%+w_7}qh*W8B9$kZ~d7M8=JbBN<mR&Q#-0#-aYW',
'lyNHKR>rZ6YZ>P<?qwXzxR`M=<7URujH?-EGwx;_&bXX$I^%Z6@r>&k=c{o)^8o(50P_UQ8!(T+yaMwK%sVg-!Mp_X6wF&NkHNeK^Bl~3Fb~4K2=gS&n=p^UybALy%)2lT!@LahG|bzmc^u|-{COVceV7MgUWj=j=8c#~VqS@PCgz=(hhkoec`D|un8#vXi+L{Qy_g4MUW|D%=FON#V_uDUHs;-!hhtujc{=9pn8#yYk9j`k{g?-2UXXc0<_(!gWL}YZM&=!vhh$!oc}nIjna5;alX*_&J(&k(UX*!K=1rMLWnPteR)5~rxhHv_+kf;vx7112HU8)O+%hFuSK&H~y6*CS(LT4eN!H1d_qqM2?sNN@WZn6{cb}U%$v%MpynSx^B>O0m_qqMI?{nLnWS`Uj{(Ww-N%m>|pWEk_Hp#xT|8<{RVv>Dp|KInytw^#@?tk6qRx!ywyZ>|h+@>elH~7DQpWEFe`yP|`x&62Ab6c5YU-Ez6KDV+-_I3aF?sGFG*=L`;&+R{TpIda2b1VP%>~p)B<eU-D9q}B}|9qcYNRo3<$@|>?<Mz2FJylpkt&bls?_sDRb)&4&(yuY}U><p{bJFK9YYTav{GCrimU<<CU*j_HT4|9=`YU!nDt#PB?twp1X2b(0lr<xye%66#)TL9d<$ygZ?<xh!A;(y$f0jn=`x;#h<F09R9QC*~{mV-IwV4M*kGAKAeQ2&u#rtc%-2e~KhB|?p+M+J06W8J@>Pu=z_l957uDOQi(4PN>dDedW6a9L1scMTommewh_`25tZ_&*;h`NNj!-u8M^V>R@Uwvp->FZrqf!*kPM51n&e!+UYkN(tZse|}kpI>nA@-fKTKfnhJBYU3~JS~ZNHQaAl4AW}d4*NG2uZ{lD#%@{QM~sW|!><_6bw@ozqy2I(@gGf4chS_s1v@rPIRrmuIy3|JV*1n{{KuTGHTtod>()VCQS+EG1H^wup^ljOxe@WBCFCCJepxEc7k@mk1nNCn)*J`E',
'T5e?({-EuEd9)T81bnx4YKD4&)_FSMwDr_4$PZb6>=E9TNpLWreh1id!01cD3%9&S{kVV!dr-eF(3o5Ja?u()geRLN4wxZ*VgnC$L4CNu*G|-r3ku1MI(tE-9>VVhb;%5V8Z`4S_@$r&HNlgDUSz|+gUzw9gWy8VP)|0v#YXtI;Mo4qkKpxNfeXQxTEedce;<tT4N2jI-w!GA4$l|TrZns}WXxaC(~vk5>^$VuXy9hZtGeK8q1qv^ztAjyLvKPWo`+uzZT|)Rm_tX$06#;QN`KhUeTAWyp*N-g4?{oag#CvZ1aHGK76l&&E4~sq8&<yt>?N#gLDY2%8`TW>7B*)e>>zAIL&O_lhq6Hr!>;7VJcd2%gK-S|{sKH9T=@<4M#DopLNCHI^+Mgh@ciq+U&6}-LSMsctb<*LH|Y*L4{zTbaY1;GL3sY~fd}Cy!+*&QJ`+Cv44yB1%4F1q4WA|PO!)jsh^xXEpF+G5zB~$c5x#OK^fi2S3~(ttu5?n#zw>+i9k0Xd@;<yTpM%fE=j3zqJ@{UHPrf(Df#bq)Qpb(s=pR>(Gsm6tz<J?3ao#wOoLA1XI`7m2zg|#Js5jIj>J{~jdZ+3k_0q4W)LZH?^_qIF>OJkiZx^%^+70c9cBR@G?apt9v`gA4?N+s8+O^-#Y4^kd;)043#0@`=5LbvZD((=6{J2D%B5o1KR9qv@`EidpNL*BLlDO%|QQ|6bR>fW7upgI+)5L8R$BFBHoG0$n56~~Deu93(??>oY=x0>FLqFvAOY~FpTl8b9U!$M<$M4-%{a~LYevy7s^_%pge!ogTOTSA$Ouwx9Y5HxyAE#fZpQqm^4<Ij4c>;NZpGS~akY|v0kcW_$kf)HhkjIeMkmr#1kOz?$ktdNikw=kNCG)Igo|VkAl6h7#&r0T5$vi8WXC?EjWS*7GvyypMGS5opS;;&rnP(;QtYn_`U&*tQKgpluPx2@Ell)2kB!7}W$)Ds;@+bL|{7L>Kf094RpZ}&$n+*_J',
'Imf6N)qGf>5BWnSWi2^(kZHQ~Y0i5{_WPH*B>A^md-}^|dZG`#((lzc$cYhhj$m25?9bmb0_W9~tBcSNUHMoI`BaVZ7vw`VQF-N@M78=j*Q4q72>D9Q)N_(2++o6bI?e3`l2>$hmOkrQ`$)g`IvbD&)DF`lFRooP9{ED;wG`6#+_MwsY;<`V;=G%#)hFb=b<>6xfjsg7d3N2mGB`(}&(a0!4*F*8kQdiaOOO1m{=`z8chTFf;JFR?wn?70OJU>*4J(>yA@2qu|7Z+uf^#0mnwgMKH%{n@^Z&+Uxobc=TgthrqTg`d!_@yx8^~=%@P4Kbx0?Mc%~3~ie!|@81I{&?;~Gky`so*(YqO*~kQuVoCOO}{_z=zyS?<2Uc_C{=R^-{Ojr(9et#ghduWh{^Fy~(x5Yz<u*?>B;mOxJ3kM)dz^Jj4WE6}i8&R13$g>{L*@u_isC-C^hj*y;1IHwj=au?181`X{l>v+5FVm%`0s~zjD!MV?-gzQ=d{j`JQnnGWK9|vGvCnRNSoGS}yP!9SOGUX5Cbwf^V!MZ?*I}h^Yp+zwKN@&lGSYHW^TPEvs_k4KYu)y0VAS;LAoMYIqy?EcSEeCO)FYH-boc9b5y^C{#;gxZWP6;1U9r@_+_3_Aahu`@EeMzCu1HDaAWGl`&rs&uU=NVJX?27e`6o-~zyi&aJVLd4#WC_+KB1(0@IzdFIMws`AX}#tDcWuFWmxu>}=qn$QFcarhQfAGIaZ6e4HJ&qNk3G=0l+%_}gxoqC`Tmrb7Z-s1v<>f>%6tvyu2Mz0kiSk<IWP1sRlClZ&s4)#Bk!JS&J&ymO0_vH`oX6<*&OF-Qr(-4^{7;zPa~h7%J*aFzcMmKD*P@oswi+FvbdZhimYB2b{5&J2G$WGI~PNL^~eF?$p1%<dWrL$k(1-F{uDW<KGtI*mwdvyQDoc@*ni~jpRo=SxvfDQ<jy#ZW8|JESkH^xZ@@X+$OCBxsB&PEzw>+i9k0Xd@;<yTpM%fE=j3zq',
'J@{UHPrf(DK^+&4lYiVejvQBxGsm6tz<J?3ao*H<<h=Ulne$FPpk7c<RK1}d`Spr=M!lmRQZH3KrQZ7Wn0ie;r{1e}K)dkU3GIe<q}mnj%x`zJL)xWkr?gwY9n-E=JEz_Iae%l$oKSItIO4|@;*5$r#34T}5vNq#B98fSjW|c#Q*n^E=*LOori!D)RX@%WcU2rFF8gtsxUJ$iaovye#C_Ed&@cG?1pS8UN9b4leujRBen|C8^izJnML(wcHTpTf-=iO-UsU}h{iffK(y!9bs(zP#*zcF=r|Gxp$5p>hKkxVZ<N@Rb<O$>r<Pj>bAkXmg4)PH467m%C7V;SK8uA?S9`YdaBJw2iCh{oqDn(K2JKeb<b)Rncle$oMyGXsL=MSZxlmqqRe1BzK1^KzFMe0-iiF$*+r&YhpWxk*up6~Jr)C=>SI^7L&zgy~FZS5;{r&edI0y!@n_4j-e8=!8PZ|DQmH}rMeBXz%;zmYm()d!$nnlHNba>#6(mPuW)_NZf+m^khy<mU*~)k?gd9`&vgPp?EB%*5?0QO_=MNk-K3N}LcS_04)8Kz+BwruR`tEV0}))XPhZx?Kb^Xg}U7!Bz^-oA9JP>eMBi$&K$PY+jAJTnV$!6oeeQ6z`YNCOhh5CRFUz0Wzv@Af&Yr>Op$H9kfB-wxDjC_dpctN_v-vp`M&K_B!g%c{_EH?^nBBL+bNoLVY%GU>(%c^L%eJ2=Z2AsT;Vj80vR<mgrC~&GXBjsGI6(H=-+K#cY=#vtC5KN{^ur>K3{`giM9JwBHK3y%Xx6x#y%t9ZmP3N2?*5Z5=Fi7-!=B-03Dtz3;^N#ihRE5!8uwomEhO%(bbD$mt(Zzth!!0qTXinskMpxk_|IolsZ0$*4o;^gUhx`6dQ+7@Zf>DpDuYTo`hG0n{6Gj+mJU**;H7$f|nOt90hcg!-(`&@ob{)R76-bv%>$c8;^9UqEg<fV!TJ1>-Sqj?qg|x6jd4D|JWf9Yg&yNAW9oFGuFWjijFH',
'H$1OBUSCG)vJQoQ+b>o@y*B&a{-_gaU+sau*r$J(D)nVcZjySmUr~R}-q-}au~(QfUh3rz!t>hGojoV@ekY=ioXzzI>b=@NN?lId!}<=$i|3#Rw!=GuA-9=PpVhYVF6PrVCl&0(HsOkVf2a%ZXY08Fb@y!T4nVJLjqAvLYP1{$S@t>VD%y&?ggx1EcR)QzTjo~K8(ZWJ)JwI6oP{0O%q392&!))*dyIE4L4Csb_$jde_-_vAP5j6Ats&pGfIY{*tOY%be|iLU>*62%f#;2X5Cyx9zn32NAAe^Na47!vJd?<f+exMRcYcq*<8^pl-iP<)bMU$NoP2KeJ@{V!_vCwX95^l<CypD(k>kp7=D2enI4_(h&Ku{E^U8VVyi*UT7pk66Z~S^hy`r8`@2H2=OX?~0mU>LRrk<;MPdo721?_}(Lp!2f(avahv_smZYNxbYza7)AY3H<i;s9}hI6>S{afG<y#~I=dafrA?oFZ-!$5dP+&iQeVI7nP1P7*hXqbjZvXZ^TK940Okr-|FdapJm)^Td6>AD~~LpP=8MAE95NpHck|{gB@;(NEED(T~xu(a+KE(GRMAk$%$eH|a;|SLtWzcj<@em+7bJx9P`KzfM2z_xt1l<OSpj<PGEz<Q3!@<Q?Q8<R#=O<Spbe<Td0u<UQm;<VEC3<W1yJ|M04$`K&S<AvCMO0~LJ-se@r2hJAlZ&|}0GO2icKNG1I+)YVa<R)WVW1!aGvQf3_X;VCuq$-c8@PV5&{x;#PLq71o-xJQ|EQ|e1BzBd(e%R|)RQI6lnzB}di@%_R>7rek@DGd<EX(B)1_nJbj5T9vktw5a?P3Nt$Z$5S$__AhM%88KsyW|ue+zoY7G=D{chigLyoP;dUN9v~3H^_dye$j{<w6p9|A7y9NRFJn)V_%{+zAxehU1WLuTUUA#>f7krmy`X16JoGGR`>g7#H+f?_pU?!l@al`KGiYo*VC8X{t&X$XNe1@4n!Rm{m#<h)B5|Bupd;PI01p7',
'A<BvUiH3Tsq+ZU@{)m4JD|<*CowF0L-_-E+2;xa&szB8BFqR*TeX+)FpYT4$*?pzX&jA<q;Tm5|k^S?QNU1kecn#tnQ|qh}FHKmCxXiT0V~4z52YO=iOh6oH&bCtaf7V%o@iY(ak2ul1Jd@O4I=v42OU)l$!x5*2rN{ncOGzu@d`tV|h`TM5iep?YTl-=^tL0`V)bp}9LNOoK^n-DKYo%$Dm+RI@_N`7ki~X_I9j_6mT5oN{{#dI$1osI@lLqlhK$(-MTNBV;2fYgz`y6?UfHjRUJ^`njBTfr=^#yr}KwT)}kie`bp?86mT!_O1J03>-92lF3_&#vOS=q09I1uj<c<(0SpuoSwppQYpmyy2+%AJ7y%RyDwV_#-a`{S|?ctmaF8-nIm$3EDg&3~YNRnVzP*smG%;4JF+1by!dJqR|8LEa`f?N8Ku3NCmB@pW)j6XN6G7V8lQ2KS0X9w&IT2k~t1tlikB8oc@(>Z}FtEQ)$R!6!V}#~OS+E$k-v>0H=r@aOi>vtZ{q<Yz+k4#edlp{EeHg`|6ieXb!n>S3Nj3IzcdLP}-8xP(-SMIIugR({mO3Tc=Rb`sKjr2M^Y3i*3S5AY+TYhl<=NRK^|$Lu{p@)LbmLZ3qV8wQI!R41ua|IY96cf1a-%lq)Yd=5SrpOeqc_uzZ+J^9`o2X$OHPX2M@IC5M$&K!5n1LuYF#ChX9a$Y&loOe|Zs26@cq25rBs8`f8>K*lvda3Fu_13S))NATF^`3S>yP%z@c0)Vz+ZFAMc1Js;U8;6UyY<^K?V5H@yC)8)xImon;|6hrxI&yE?x;9KT=L@-af>)cTvKt5xaY?~;v#X9xJevUag{jh$6ewuahW)+;x=*IkL$#F;y(R=>KEuI{C<Oegnos7hJHu&L-b32KSjSqKSsYsKd1UV`a!>6q@SeUq#vbUrJq&(F8#3IFVj!cZ_|&{uhY-d?~@0pynsBx&l|`i$ScS*$UDeG$V<pm$Xm!`$ZN=R$a~0x$cxC6',
'$eYNc$gBMMtY=cc-Iw~3)Hk0qMCzW`IwEDvYfgiVpL0>_r{`!U{4NN*-S@WULAlJ++)}6gn()jUnZT=k=kK7tzVG;7=wIR6H}aV9$^pU`;~t@&zi;7TsmnesqKMR?|15RX2UeE4^j$JG6#hCMeB4)a0OCDgsjtE_^LW7Ded$&U-wb$+el&@my#d0L^JbU2{8gpS|3w}8p(GyYg?^cd8^*kaoL>>+k~nT}TFAan5jP~ZK0aT{zPCX?i^QV)@LY+RZkCWf1v{l5f}_Q6kgv|8Pfo%$gY+{z6efKb*8RZyC(K$8o}DnV5c(4&bY5H%vfd-aPYERwF)j&N^%7?U{6HTLukFlP$min`A9*k4MO^0HeHQ&Jyek?>KZ!{<(U-&9zb5(%d0QSr-xzP@?3hPy-dX5-;*D@3uJ9ywkv<&n?;`H-+^moO0G`9w5m$QFw?$m+ndv~?eb4YU=ojLNX@~v-o;vxrKt`vKcr8;H;uMc1PiDk-f47(X+0&hfXWSREPLX~h`;k|0uk0&*N~Y9C{|5Kq2IyDhZa*6O=&p4K^X`srCHc23f!n1I%6I9T;?kp!h3joy^tW)`yokObuH&O!(s!i?`VqL6%*4FACVj<ty9Q5{@5gjO-waoSF$W;aJ;(!@Z^&MWkDFpXT-K53ALMks$qD&!5#m1Q!#M{fzP^L|IS+S%{y4W*fc`jFbVoeqocT<?KXyId&pGfg`b;>xWWhW;n?Dv=>!bA7DOdXgWMK`)*BKRzJc%>SAc^OL^Gp96U8S|i4>)&xgZws3`u)6FB7KD(r$QX;xcMD@WgHiCps$DH<YA11<KQ;L^NyWf=(%I_HslQ)YmZ3Zq7`|hAKpSY`dB&U<dN%4I|_YtOxlV58;<eDI+D*g{)_ZAde{nbXbZ%VjzJ~SXTi~bG~!Z6-|WZ>IC>ZA4B0ad_Uz~}So$$_-`ZVdjY_JlnB?#L9)HK{@VdMY@5|@lbMZO(+<Xtd7vGcb&2iwkaGW@997m2T$C=~KdEmTo',
'p455cJo@LA^UQgt9#Ai+C)6A25%r3CM!lmRQZK2e)LZH?^;*?)>b>6%Xcx2-+70c9c11g*-O&zdm$Xybt!l@#Yrmb-?ui4$1>yv8gE&H5A<htYR2(8M`EiQ4MI0lp5$A||#6jXBagw;H;wW*|kF&&G;xKWUI8EFpjuY33^D6Gs5BU88{RI66{RsUE{S5sM{Sf^U{gmpr=*Rqijed@PkA9GTk$#eXlYW$bm423fmws6F%k<NJzfC_*zfM0-zfT@OUO=8e-asBfUO}Eg-a#HhUP7Ki-a;NjUPGQk-a{TlUPPWm-b5bt53fqfvu^K`eAi{UuJ@$e-*rIx-PraA%Y9DCxO{pleSO}90ndC-I@;tiH_@licd3W;qdA%TfxKTj>A$nNvW&-ywa|0l>`KxfXM8r;nQu^i>37rVtc+9RS?H_ftF!}oUtghIz$ITst>pg#Dgp-*9q(Y@iLc!lhs5h+Wj+u80z64v|06{DC%pk)C5~u=e0O3@E8s+8t=F*Y#KH-Z_e`?|`RzpI2lBZI?{DrEy=>ASa_>m#54EfY^gm(jDXaq|bonfOt!iCD{yd>T2dDJ8+J*P`+E2soy^j{*{@&w-fe+p_i=+?N#EZC&xBE4*)7qQF4)eEx9e6`NV|~E$vo-oxdG2gOzd6tTx6otHQj_GJM@4RzymTb=%TpoX4P<6J?8~EljQMuIIflMZ?sIX#WA~=Xcz^e_LFmWj?$aZs^k?fOI96yB#@8Lb3i)hT{C()7>wZq;zg>ssNj`o>TC88VVowWx#>^ZCS#^x`jmx>vBe?wl&*A)56MeRvH{WBu!@2Je^3Kks$AK%(F&^|Qb9Nf=Rr>g4MZVjaFBExOXIKS1kHfwb{hb_7x}lGp<8&wBh-1_5!art5qmh?mSQzf>XjfMJYSjbMk1*dPjDsU$<7(-{m>D=@fBQrFLtamZ^$7dHH7}%3<RsXceZ~b@pBdg8^Jnig5ZATWdjx#5m)Z=xu}3{0A$>f@0H<ueNf>9_7bp5_',
'*&f|@$@)+faKv`tCEm-nF&Ou;E!u|pwM{+-y|N9j48LsaQ5O9cZLOqlp{>q*_+MMOzhSSoLYINNw(J?&%DUGl=%FpJDD1+fIBNku?4{u+;y-(!=kc#fUzc?_7y1{*-zW*c6Mx=`JbL`eQs4{m2R*RU_}vw;P7}Y)3|xr+y@{;ft;vG?eEf<bSa*tFQX76Ge!=g+&-l5Mp-1tvJ_9%6XWW(FPp>N9n^p>X8b5V!b5%A=@^^lZzvFdyUEYWH<#X`4_?&!hz6aln@5%S(IB;AzP8>IN967H3apt&l9yl+YC(aw^k@L!V=Dbr6s29`|>J9aXdZp?a_0F${)Jy6q^_F@}y{4X1?`a3L3)N0&H-0;!UD3{HceF#=CGC`UtJ*Q`+HdEyd*T3bfjB|jAdaZGLY(pA4snRMM4TdS5yyyYD$Wu2{5VKlBu)}HiKE0-6=#XNejFw)6Q_yW#Bt&}abCrJ`T@URpr4@MpdX=Mp`W4OQT-78lHX6!Z_$s@uhGxZ@6ivcevy9C?>Fg3=~wAz>38Xe>6huJ>9^^}RliO@@Av!U0ptbb3FHmr5#$x*8RQ-0A><|GDda8WG2}JmIpjU$LF7f`N#srBQUCC&f3sW<PgMLj|E<|{fBwIDF5UI&dq({G?+Uq^{=1(~+vjCH)PcBMph<rDcU9~cy}vMv6Y0VDUZ``w^+lFqNMi<G`8U6)UH9tyzxit2x;no;`uBQIzfAl$Z>wK(+WGH3K5fSdU;gcTpgZ{Z^1t~*&4+}Z|Mut5UWpl5^51{=Ut2!){{tq^$T$',
))

def _fail_translation_canonical_blob(reason: str) -> RuntimeError:
    return RuntimeError(f"invalid canonical translation blob: {reason}")


def _decode_translation_canonical_blob() -> tuple[tuple[int, ...], ...]:
    """Decode and authenticate the bounded canonical int64 payload."""

    encoded = _TRANSLATION_CANONICAL_BLOB_B85.encode("ascii")
    if len(encoded) != _TRANSLATION_CANONICAL_BLOB_B85_LENGTH:
        raise _fail_translation_canonical_blob("encoded byte count")
    if hashlib.sha256(encoded).hexdigest() != TRANSLATION_CANONICAL_BLOB_SHA256:
        raise _fail_translation_canonical_blob("encoded SHA-256")
    try:
        compressed = base64.b85decode(encoded)
    except (ValueError, TypeError) as exc:
        raise _fail_translation_canonical_blob("base85 decoding") from exc
    if len(compressed) != _TRANSLATION_CANONICAL_COMPRESSED_BYTES:
        raise _fail_translation_canonical_blob("compressed byte count")
    if (
        hashlib.sha256(compressed).hexdigest()
        != TRANSLATION_CANONICAL_COMPRESSED_SHA256
    ):
        raise _fail_translation_canonical_blob("compressed SHA-256")

    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(
            compressed, _TRANSLATION_CANONICAL_RAW_BYTES + 1
        )
    except zlib.error as exc:
        raise _fail_translation_canonical_blob("zlib decoding") from exc
    if len(raw) != _TRANSLATION_CANONICAL_RAW_BYTES:
        raise _fail_translation_canonical_blob("decompressed byte count")
    if not inflater.eof or inflater.unconsumed_tail or inflater.unused_data:
        raise _fail_translation_canonical_blob("non-canonical zlib stream")
    if hashlib.sha256(raw).hexdigest() != TRANSLATION_CANONICAL_RAW_SHA256:
        raise _fail_translation_canonical_blob("decompressed SHA-256")

    header_size = struct.calcsize(_TRANSLATION_CANONICAL_HEADER_FORMAT)
    try:
        header = struct.unpack(
            _TRANSLATION_CANONICAL_HEADER_FORMAT, raw[:header_size]
        )
    except struct.error as exc:
        raise _fail_translation_canonical_blob("header") from exc
    magic, version, scale, *counts_and_total = header
    section_counts = tuple(counts_and_total[:-1])
    total_elements = counts_and_total[-1]
    if magic != _TRANSLATION_CANONICAL_BLOB_MAGIC:
        raise _fail_translation_canonical_blob("magic")
    if version != TRANSLATION_CANONICAL_BLOB_VERSION:
        raise _fail_translation_canonical_blob("version")
    if scale != TRANSLATION_SCHEDULE_QUANTIZATION_SCALE:
        raise _fail_translation_canonical_blob("scale")
    if section_counts != _TRANSLATION_CANONICAL_SECTION_COUNTS:
        raise _fail_translation_canonical_blob("section shapes")
    if total_elements != sum(section_counts):
        raise _fail_translation_canonical_blob("element count")
    payload = raw[header_size:]
    if len(payload) != total_elements * 8:
        raise _fail_translation_canonical_blob("payload byte count")
    try:
        encoded_values = tuple(
            item[0] for item in struct.iter_unpack(">q", payload)
        )
    except struct.error as exc:
        raise _fail_translation_canonical_blob("int64 payload") from exc
    if len(encoded_values) != total_elements:
        raise _fail_translation_canonical_blob("decoded element count")

    sections: list[tuple[int, ...]] = []
    cursor = 0
    for count in section_counts:
        sections.append(encoded_values[cursor : cursor + count])
        cursor += count
    if cursor != total_elements:
        raise _fail_translation_canonical_blob("section boundary")
    return tuple(sections)


def _restore_delta_rows(
    encoded: tuple[int, ...], rows: int, columns: int
) -> tuple[tuple[int, ...], ...]:
    if len(encoded) != rows * columns:
        raise _fail_translation_canonical_blob("delta row shape")
    restored: list[tuple[int, ...]] = []
    cursor = 0
    for _ in range(rows):
        value = 0
        row: list[int] = []
        for delta in encoded[cursor : cursor + columns]:
            value += delta
            if not -(1 << 63) <= value < (1 << 63):
                raise _fail_translation_canonical_blob("delta overflow")
            row.append(value)
        restored.append(tuple(row))
        cursor += columns
    return tuple(restored)


def _restore_delta_xy(
    encoded: tuple[int, ...], rows: int, points: int
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if len(encoded) != rows * points * 2:
        raise _fail_translation_canonical_blob("XY section shape")
    restored: list[tuple[tuple[int, int], ...]] = []
    cursor = 0
    for _ in range(rows):
        x = _restore_delta_rows(encoded[cursor : cursor + points], 1, points)[0]
        cursor += points
        y = _restore_delta_rows(encoded[cursor : cursor + points], 1, points)[0]
        cursor += points
        restored.append(tuple(zip(x, y)))
    return tuple(restored)


def _canonical_int64_sha256(values: Iterable[int]) -> str:
    flat = tuple(values)
    try:
        packed = struct.pack(f">{len(flat)}q", *flat)
    except struct.error as exc:
        raise _fail_translation_canonical_blob("absolute int64 grid") from exc
    return hashlib.sha256(packed).hexdigest()


def _grid_to_float_rows(
    rows: tuple[tuple[int, ...], ...]
) -> tuple[tuple[float, ...], ...]:
    scale = TRANSLATION_SCHEDULE_QUANTIZATION_SCALE
    return tuple(tuple(value / scale for value in row) for row in rows)


def _xy_grid_to_float_rows(
    rows: tuple[tuple[tuple[int, int], ...], ...]
) -> tuple[tuple[tuple[float, float], ...], ...]:
    scale = TRANSLATION_SCHEDULE_QUANTIZATION_SCALE
    return tuple(
        tuple((x / scale, y / scale) for x, y in row) for row in rows
    )


(
    _route_progress_delta,
    _lead_progress_delta,
    _anchor_xy_delta,
    _route_xy_delta,
    _lead_xy_delta,
    _route_peaks_absolute,
    _lead_peaks_absolute,
) = _decode_translation_canonical_blob()

_route_progress_grid = _restore_delta_rows(
    _route_progress_delta, 5, TRANSLATION_PATH_ROUTE_SAMPLES + 1
)
_lead_progress_grid = _restore_delta_rows(
    _lead_progress_delta, 5, TRANSLATION_PATH_LEAD_IN_TICKS + 1
)
_anchor_xy_grid = _restore_delta_xy(
    _anchor_xy_delta, 5, TRANSLATION_PATH_WAYPOINTS
)
_route_xy_grid = _restore_delta_xy(
    _route_xy_delta, 5, TRANSLATION_PATH_ROUTE_SAMPLES + 1
)
_lead_xy_grid = _restore_delta_xy(
    _lead_xy_delta, 5, TRANSLATION_PATH_LEAD_IN_TICKS + 1
)
_route_peaks_grid = tuple(
    tuple(_route_peaks_absolute[index : index + 2])
    for index in range(0, len(_route_peaks_absolute), 2)
)
_lead_peaks_grid = tuple(
    tuple(_lead_peaks_absolute[index : index + 2])
    for index in range(0, len(_lead_peaks_absolute), 2)
)

_canonical_section_hashes = (
    _canonical_int64_sha256(value for row in _route_progress_grid for value in row),
    _canonical_int64_sha256(value for row in _lead_progress_grid for value in row),
    _canonical_int64_sha256(
        value for row in _anchor_xy_grid for point in row for value in point
    ),
    _canonical_int64_sha256(
        value for row in _route_xy_grid for point in row for value in point
    ),
    _canonical_int64_sha256(
        value for row in _lead_xy_grid for point in row for value in point
    ),
    _canonical_int64_sha256(
        value for row in _route_peaks_grid for value in row
    ),
    _canonical_int64_sha256(
        value for row in _lead_peaks_grid for value in row
    ),
)
if _canonical_section_hashes != (
    TRANSLATION_ROUTE_PROGRESS_TABLE_SHA256,
    TRANSLATION_LEAD_PROGRESS_TABLE_SHA256,
    TRANSLATION_PATH_WAYPOINTS_XY_SHA256,
    TRANSLATION_ROUTE_TARGETS_XY_SHA256,
    TRANSLATION_LEAD_OFFSETS_XY_SHA256,
    TRANSLATION_ROUTE_KINEMATIC_PEAKS_SHA256,
    TRANSLATION_LEAD_KINEMATIC_PEAKS_SHA256,
):
    raise _fail_translation_canonical_blob("section SHA-256")

TRANSLATION_ROUTE_PROGRESS_TABLE = _grid_to_float_rows(_route_progress_grid)
TRANSLATION_LEAD_PROGRESS_TABLE = _grid_to_float_rows(_lead_progress_grid)
TRANSLATION_PATH_WAYPOINTS_XY_M = _xy_grid_to_float_rows(_anchor_xy_grid)
TRANSLATION_ROUTE_TARGETS_XY_M = _xy_grid_to_float_rows(_route_xy_grid)
TRANSLATION_LEAD_OFFSETS_XY_M = _xy_grid_to_float_rows(_lead_xy_grid)
TRANSLATION_ROUTE_KINEMATIC_PEAKS = _grid_to_float_rows(_route_peaks_grid)
TRANSLATION_LEAD_KINEMATIC_PEAKS = _grid_to_float_rows(_lead_peaks_grid)

ROTATION_MOTION_TICKS = 2_400
ROTATION_TERMINAL_HOLD_TICKS = 1_802
ROTATION_FORMAL_ACTION_TICKS = 4_202
FORMAL_PRE_ACTION_TICKS = 298
FORMAL_EPISODE_TICKS = 4_500
ROTATION_FRAME = "object_local_z"
ROTATION_AXIS_LOCAL = (0.0, 0.0, 1.0)
# Rotation difficulty is an episode goal, not an endpoint-success tolerance.
# Endpoint attitude success is checked independently on wrapped fixed-XYZ
# Euler errors.  Both curriculum intervals are closed integer intervals.
ROTATION_ANGLE_MIN_DEG = 10
ROTATION_ANGLE_MAX_DEG = 45
# Read-only compatibility aliases for pre-v19 artifact readers.
PHASE1_ROTATION_ANGLE_MIN_DEG = ROTATION_ANGLE_MIN_DEG
PHASE1_ROTATION_ANGLE_MAX_DEG = ROTATION_ANGLE_MAX_DEG
PHASE2_ROTATION_ANGLE_MIN_DEG = ROTATION_ANGLE_MIN_DEG
PHASE2_ROTATION_ANGLE_MAX_DEG = ROTATION_ANGLE_MAX_DEG
# Retained as the maximum requested rotation for compatibility with analysis
# code that needs one scalar plotting bound; environment goals are per-world.
ROTATION_ANGLE_DEG = 45.0

ALGORITHM = "ActionChunkDiffusionIKImitation"
PHYSICS_BACKEND = "native-mujoco-warp"
TORCH_WARP_EXCHANGE = "persistent-zero-copy-alias"
CONTROLLERS = ("closed_loop_ik_expert", "ik_diffusion")
TOTAL_TRAINING_STEPS = 90_000_000
SUCCESS_GATE_ENABLED = True

# Formal RL physics uses the MJWarp-supported dense Newton path.  The XML
# requests MuJoCo's strict 1e-10 tolerance, while the pinned float32 MJWarp
# runtime has an effective 1e-6 floor.  Dense Newton is supported only through
# nv=60, so the RL XML excludes the untrained plate free body and must remain
# at or below that limit on both grippers.
FORMAL_SOLVER = "Newton"
FORMAL_JACOBIAN = "dense"
FORMAL_XML_SOLVER_TOLERANCE = 1.0e-10
FORMAL_EFFECTIVE_SOLVER_TOLERANCE = 1.0e-6
FORMAL_DENSE_NEWTON_MAX_NV = 60
RL_XML_EXCLUDES_UNTRAINED_PLATE = True
PHYSICS_LAYOUT = "task_specific_four_single_object_xml_shards"
SHARD_OBJECT_ORDER = OBJECTS
OBJECT_SHARD_COUNT = len(SHARD_OBJECT_ORDER)
ACTIVE_FREE_OBJECTS_PER_SHARD = 1
CUDA_GRAPHS_PER_GLOBAL_BATCH = OBJECT_SHARD_COUNT
FORMAL_GLOBAL_WORLDS = 20_000
FORMAL_OBJECT_WORLD_COUNTS = (5_000,) * OBJECT_SHARD_COUNT
FORMAL_REMAINDER_WORLDS = 0
FORMAL_REMAINDER_OBJECT_WORLD_COUNTS = (0,) * OBJECT_SHARD_COUNT
# Read-only compatibility aliases for pre-v19 artifact readers.
FORMAL_PHASE_REMAINDER_WORLDS = FORMAL_REMAINDER_WORLDS
FORMAL_PHASE1_REMAINDER_OBJECT_WORLD_COUNTS = (
    FORMAL_REMAINDER_OBJECT_WORLD_COUNTS
)
FORMAL_PHASE2_REMAINDER_OBJECT_WORLD_COUNTS = (
    FORMAL_REMAINDER_OBJECT_WORLD_COUNTS
)

# Each parent-controller reset produces one physically distinct deterministic
# start state and analytic grasp plan.  Training samples this bank uniformly;
# paired evaluation assigns one unique template to each of its 32 repeats.
RESET_TEMPLATES_PER_OBJECT = 32
RESET_TEMPLATE_SEED_BASE = 10_000
RESET_TEMPLATE_OBJECT_STRIDE = 1_000

TIP_ORDER = ("U", "D", "L", "R")
XYZ_ORDER = ("x", "y", "z")
ACTION_DIM = 25

TIP_HEIGHT_SLICE = slice(0, 4)
YAW_SLICE = slice(4, 5)
YAW_INDEX = 4
TIP_PRELOAD_SLICE = slice(5, 9)
TIP_XYZ_RESIDUAL_SLICE = slice(9, 21)
FORCE_SCALE_SLICE = slice(21, 25)
FORCE_SCALE_INDICES = tuple(range(21, 25))

EPISODE_LEVEL_ACTION_SLICES = (
    TIP_HEIGHT_SLICE,
    YAW_SLICE,
    TIP_PRELOAD_SLICE,
    FORCE_SCALE_SLICE,
)
EPISODE_LEVEL_ACTION_INDICES = tuple(range(0, 9)) + FORCE_SCALE_INDICES
DYNAMIC_ACTION_SLICE = TIP_XYZ_RESIDUAL_SLICE
DYNAMIC_ACTION_INDICES = tuple(range(9, 21))

NORMALIZED_ACTION_MIN = -1.0
NORMALIZED_ACTION_MAX = 1.0
ACTION_LOW = (NORMALIZED_ACTION_MIN,) * ACTION_DIM
ACTION_HIGH = (NORMALIZED_ACTION_MAX,) * ACTION_DIM
ZERO_ACTION = (0.0,) * ACTION_DIM

GRASP_PLANE_MIN_FRACTION = 0.40
GRASP_PLANE_NOMINAL_FRACTION = 0.50
GRASP_PLANE_MAX_FRACTION = 0.60
FIXED_GRASP_PLANE_FRACTIONS: dict[str, float] = {}
LEARNED_GRASP_PLANE_OBJECTS = frozenset(FRUIT_OBJECTS)
FIXED_GRASP_PLANE_YAW_RAD = 0.0
FIXED_GRASP_PLANE_PHYSICAL_TOLERANCE_M = 0.002
YAW_LIMIT_DEG = 90.0
YAW_LIMIT_RAD = math.radians(YAW_LIMIT_DEG)
# ``YAW_LIMIT_RAD`` scales the policy's *residual* yaw command.  The canonical
# geometry planner, however, freezes an absolute principal-angle orientation
# into the hard-tip anchors and legitimately searches 0--165 degrees.  Keep
# those two domains distinct: rejecting a 105--165 degree frozen planner yaw
# as though it were a policy residual makes otherwise valid G6 templates
# impossible to load.
FROZEN_PLANNER_YAW_ABS_LIMIT_RAD = math.pi
TIP_PRELOAD_LIMIT_M = 0.003
TIP_XYZ_RESIDUAL_LIMIT_M = 0.003
FORCE_SCALE_MIN = 0.5
FORCE_SCALE_NOMINAL = 1.0
FORCE_SCALE_MAX = 1.5

# Endpoint success is an axis-aligned box in pose-error space.  L2 norms remain
# useful diagnostics and dense-reward inputs, but they never classify success.
# All active objects are fruits and therefore share one threshold per task.
TRANSLATION_POSITION_AXIS_ABS_TOLERANCE_M = 0.010
TRANSLATION_EULER_AXIS_ABS_TOLERANCE_DEG = 20.0
# A translation demonstration must track the entire frozen route, not merely
# return to a closed path's endpoint.  The certificate is evaluated per world
# and per Cartesian axis over every active path-following control tick.
TRANSLATION_PATH_TRACKING_AXIS_ABS_TOLERANCE_M = 0.010
ROTATION_POSITION_AXIS_ABS_TOLERANCE_M = 0.010
ROTATION_EULER_AXIS_ABS_TOLERANCE_DEG = 10.0

# A demonstration is certified only after these physical conditions remain
# true for a contiguous terminal window.  Keep the values in the canonical
# contract so capacity reports and formal artifacts cannot silently bind a
# controller with different expert gates.
EXPERT_STABLE_TICKS = 25
# Lift admission is shorter than the terminal certificate but is evaluated
# continuously through the exact lift boundary.
PRELIFT_CONTACT_STABLE_TICKS = 5
EXPERT_OBJECT_LINEAR_SPEED_MAX_M_S = 0.005
EXPERT_OBJECT_ANGULAR_SPEED_MAX_RAD_S = 0.05
EXPERT_OBJECT_TABLE_CLEARANCE_M = 0.005
EXPERT_MIN_CONTACTS = 3
EXPERT_PREFERRED_CONTACTS = 4
TIP_TARGET_SPEED_MAX_M_S = 0.040
TIP_TARGET_ACCELERATION_MAX_M_S2 = 0.150
TIP_TARGET_JERK_MAX_M_S3 = 1.0
IK_CAGE_TARGET_ERROR_MAX_M = 0.003

# A finite active object is terminally dropped once the conservative lower
# bound used by the parent IK-only controllers crosses 10 mm below the table.
# The terminal is sticky, but formal vector batches retain their fixed shape
# and horizon through an absorbing, quarantined continuation.
TABLE_TOP_Z_M = 0.260
# Every G6 Diffusion episode starts with the active fruit COM at this world
# height.  Keeping the value in the signed algorithm contract prevents a
# capacity report or checkpoint built with the former table-rest reset from
# being admitted silently.
INITIAL_OBJECT_COM_Z_M = 0.340
DROP_MARGIN_M = 0.010
OBJECT_DROP_THRESHOLD_Z_M = TABLE_TOP_Z_M - DROP_MARGIN_M
OBJECT_DROP_TERMINAL_REWARD = -10.0
OBJECT_DROP_QUARANTINE_POSITION_M = (2.0, 2.0, 1.0)


def reset_template_seed(object_index: int, template_index: int) -> int:
    """Return the stable parent-reset seed for one object/template pair."""

    if not 0 <= int(object_index) < len(OBJECTS):
        raise ValueError("object_index is outside the training object set")
    if not 0 <= int(template_index) < RESET_TEMPLATES_PER_OBJECT:
        raise ValueError("template_index is outside the reset template bank")
    return (
        RESET_TEMPLATE_SEED_BASE
        + int(object_index) * RESET_TEMPLATE_OBJECT_STRIDE
        + int(template_index)
    )


@dataclass(frozen=True)
class ActionField:
    """One contiguous part of the normalized policy action."""

    name: str
    start: int
    stop: int
    shape: tuple[int, ...]
    scope: str
    applicability: str
    physical_unit: str
    physical_min: float
    physical_max: float

    @property
    def sl(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def size(self) -> int:
        return self.stop - self.start


ACTION_FIELDS = (
    ActionField(
        "tip_height",
        0,
        4,
        (4,),
        "episode",
        "fruit_only",
        "object_height_fraction",
        GRASP_PLANE_MIN_FRACTION,
        GRASP_PLANE_MAX_FRACTION,
    ),
    ActionField(
        "yaw",
        4,
        5,
        (1,),
        "episode",
        "fruit_only",
        "rad",
        -YAW_LIMIT_RAD,
        YAW_LIMIT_RAD,
    ),
    ActionField(
        "tip_preload",
        5,
        9,
        (4,),
        "episode",
        "all_objects",
        "m",
        -TIP_PRELOAD_LIMIT_M,
        TIP_PRELOAD_LIMIT_M,
    ),
    ActionField(
        "tip_xyz_residual",
        9,
        21,
        (4, 3),
        "control_tick",
        "all_objects",
        "m",
        -TIP_XYZ_RESIDUAL_LIMIT_M,
        TIP_XYZ_RESIDUAL_LIMIT_M,
    ),
    ActionField(
        "force_scale",
        21,
        25,
        (4,),
        "episode",
        "all_objects",
        "nominal_multiplier",
        FORCE_SCALE_MIN,
        FORCE_SCALE_MAX,
    ),
)

ACTION_SLICES = {field.name: field.sl for field in ACTION_FIELDS}
ACTION_LIMITS = {
    field.name: (field.physical_min, field.physical_max)
    for field in ACTION_FIELDS
}


@dataclass(frozen=True)
class DecodedAction:
    """Physical action values after object applicability has been applied."""

    tip_height_fraction: tuple[float, float, float, float]
    yaw_rad: float
    tip_preload_m: tuple[float, float, float, float]
    tip_xyz_residual_m: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    force_scale: tuple[float, float, float, float]


def validate_object(object_kind: str) -> str:
    value = str(object_kind)
    if value not in OBJECTS:
        raise ValueError(f"unsupported training object: {value!r}")
    return value


def validate_task(task: str) -> str:
    value = str(task)
    if value not in TASKS:
        raise ValueError(f"unsupported task: {value!r}")
    return value


def grasp_plane_is_learned(object_kind: str) -> bool:
    """Return whether RL selects grasp-plane height and yaw for this object."""

    return validate_object(object_kind) in LEARNED_GRASP_PLANE_OBJECTS


def grasp_plane_nominal_fraction(object_kind: str) -> float:
    """Return the object-local nominal mean hard-tip height fraction."""

    kind = validate_object(object_kind)
    if kind in FIXED_GRASP_PLANE_FRACTIONS:
        return float(FIXED_GRASP_PLANE_FRACTIONS[kind])
    return GRASP_PLANE_NOMINAL_FRACTION


def grasp_plane_fraction_limits(object_kind: str) -> tuple[float, float]:
    """Return commanded mean-height limits for one training object."""

    kind = validate_object(object_kind)
    if kind in LEARNED_GRASP_PLANE_OBJECTS:
        return GRASP_PLANE_MIN_FRACTION, GRASP_PLANE_MAX_FRACTION
    fixed = grasp_plane_nominal_fraction(kind)
    return fixed, fixed


def validate_action(action: Iterable[object]) -> tuple[float, ...]:
    """Return a finite normalized 25-D action or raise ``ValueError``."""

    if isinstance(action, (str, bytes)):
        raise ValueError("action must be a numeric iterable")
    try:
        values = tuple(float(value) for value in action)
    except (TypeError, ValueError) as exc:
        raise ValueError("action must be a numeric iterable") from exc
    if len(values) != ACTION_DIM:
        raise ValueError(
            f"action must contain exactly {ACTION_DIM} values, got {len(values)}"
        )
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"action[{index}] is not finite")
        if value < NORMALIZED_ACTION_MIN or value > NORMALIZED_ACTION_MAX:
            raise ValueError(
                f"action[{index}]={value!r} is outside "
                f"[{NORMALIZED_ACTION_MIN}, {NORMALIZED_ACTION_MAX}]"
            )
    return values


def mask_action_for_object(
    action: Iterable[object],
    object_kind: str,
) -> tuple[float, ...]:
    """Validate and zero channels that cannot affect the selected object."""

    kind = validate_object(object_kind)
    values = list(validate_action(action))
    if kind not in LEARNED_GRASP_PLANE_OBJECTS:
        values[TIP_HEIGHT_SLICE] = (0.0,) * len(TIP_ORDER)
        values[YAW_INDEX] = 0.0
    return tuple(values)


def episode_level_action(
    action: Iterable[object],
    object_kind: str,
) -> tuple[float, ...]:
    values = mask_action_for_object(action, object_kind)
    return tuple(values[index] for index in EPISODE_LEVEL_ACTION_INDICES)


def dynamic_action(action: Iterable[object]) -> tuple[float, ...]:
    values = validate_action(action)
    return values[DYNAMIC_ACTION_SLICE]


def validate_episode_action_stability(
    episode_action: Iterable[object],
    tick_action: Iterable[object],
    object_kind: str,
    *,
    atol: float = 0.0,
) -> None:
    """Require episode-scoped channels to remain fixed across control ticks."""

    tolerance = float(atol)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("atol must be finite and non-negative")
    expected = episode_level_action(episode_action, object_kind)
    actual = episode_level_action(tick_action, object_kind)
    if any(abs(left - right) > tolerance for left, right in zip(expected, actual)):
        raise ValueError("episode-level action channels changed within an episode")


def decode_action(
    action: Iterable[object],
    object_kind: str,
) -> DecodedAction:
    """Map normalized action values to the physical residual contract."""

    kind = validate_object(object_kind)
    values = mask_action_for_object(action, kind)
    if grasp_plane_is_learned(kind):
        height_fractions = tuple(
            GRASP_PLANE_NOMINAL_FRACTION
            + value
            * (
                GRASP_PLANE_NOMINAL_FRACTION - GRASP_PLANE_MIN_FRACTION
                if value <= 0.0
                else GRASP_PLANE_MAX_FRACTION
                - GRASP_PLANE_NOMINAL_FRACTION
            )
            for value in values[TIP_HEIGHT_SLICE]
        )
        yaw_rad = values[YAW_INDEX] * YAW_LIMIT_RAD
    else:
        height_fractions = (grasp_plane_nominal_fraction(kind),) * len(
            TIP_ORDER
        )
        yaw_rad = FIXED_GRASP_PLANE_YAW_RAD
    preloads = tuple(
        value * TIP_PRELOAD_LIMIT_M for value in values[TIP_PRELOAD_SLICE]
    )
    flat_xyz = tuple(
        value * TIP_XYZ_RESIDUAL_LIMIT_M
        for value in values[TIP_XYZ_RESIDUAL_SLICE]
    )
    xyz = tuple(
        tuple(flat_xyz[3 * tip : 3 * tip + 3])
        for tip in range(len(TIP_ORDER))
    )
    force_scale = tuple(
        FORCE_SCALE_NOMINAL
        + value * (FORCE_SCALE_MAX - FORCE_SCALE_MIN) / 2.0
        for value in values[FORCE_SCALE_SLICE]
    )
    return DecodedAction(
        tip_height_fraction=height_fractions,  # type: ignore[arg-type]
        yaw_rad=yaw_rad,
        tip_preload_m=preloads,  # type: ignore[arg-type]
        tip_xyz_residual_m=xyz,  # type: ignore[arg-type]
        force_scale=force_scale,
    )


def translation_path_contract_payload() -> dict[str, object]:
    """Return the canonical five-path geometry and timing contract."""

    return {
        "frame": TRANSLATION_FRAME,
        "dz_m": _quantize_translation_schedule_value(TRANSLATION_DZ_M),
        "reference_capture": ACTION_REFERENCE_CAPTURE,
        "path_family": TRANSLATION_PATH_FAMILY,
        "region_radius_m": _quantize_translation_schedule_value(
            TRANSLATION_PATH_REGION_RADIUS_M
        ),
        "segments": TRANSLATION_PATH_SEGMENTS,
        "waypoints": TRANSLATION_PATH_WAYPOINTS,
        "shape_names": TRANSLATION_PATH_SHAPES,
        "waypoints_xy_sha256": TRANSLATION_PATH_WAYPOINTS_XY_SHA256,
        "canonical_grid_hash_encoding": (
            TRANSLATION_SCHEDULE_TABLE_HASH_ENCODING
        ),
        "path_seed_provenance_sentinel": (
            TRANSLATION_PATH_SEED_MIN,
            TRANSLATION_PATH_SEED_MAX,
        ),
        "path_start_radius_max_m": _quantize_translation_schedule_value(
            TRANSLATION_PATH_START_RADIUS_MAX_M
        ),
        "relative_extent_max_m": _quantize_translation_schedule_value(
            TRANSLATION_PATH_RELATIVE_EXTENT_MAX_M
        ),
        "generation_radius_max_m": _quantize_translation_schedule_value(
            TRANSLATION_PATH_GENERATION_RADIUS_M
        ),
        "arc_step_m_by_shape": tuple(
            _quantize_translation_schedule_value(value)
            for value in TRANSLATION_PATH_STEP_M_BY_SHAPE
        ),
        "arc_length_m_by_shape": tuple(
            _quantize_translation_schedule_value(value)
            for value in TRANSLATION_PATH_LENGTH_M_BY_SHAPE
        ),
        "maximum_arc_step_m": _quantize_translation_schedule_value(
            TRANSLATION_PATH_MAX_STEP_M
        ),
        "maximum_arc_length_m": _quantize_translation_schedule_value(
            TRANSLATION_PATH_MAX_LENGTH_M
        ),
        "timing_control_ticks": {
            "lead_in": TRANSLATION_PATH_LEAD_IN_TICKS,
            "route_start_settle": TRANSLATION_PATH_START_SETTLE_TICKS,
            "strict_route_samples": TRANSLATION_PATH_ROUTE_SAMPLES,
            "terminal_hold": TRANSLATION_PATH_TERMINAL_HOLD_TICKS,
            "per_segment": TRANSLATION_PATH_SEGMENT_TICKS,
            "translation_formal_action": FORMAL_ACTION_TICKS,
            "rotation_formal_action": ROTATION_FORMAL_ACTION_TICKS,
        },
        "bounded_schedule": {
            "quantization_decimals": (
                TRANSLATION_SCHEDULE_QUANTIZATION_DECIMALS
            ),
            "quantization_scale": TRANSLATION_SCHEDULE_QUANTIZATION_SCALE,
            "progress_table_hash_encoding": (
                TRANSLATION_SCHEDULE_TABLE_HASH_ENCODING
            ),
            "canonical_blob": {
                "encoding": TRANSLATION_CANONICAL_BLOB_ENCODING,
                "version": TRANSLATION_CANONICAL_BLOB_VERSION,
                "encoded_sha256": TRANSLATION_CANONICAL_BLOB_SHA256,
                "compressed_sha256": (
                    TRANSLATION_CANONICAL_COMPRESSED_SHA256
                ),
                "decompressed_sha256": TRANSLATION_CANONICAL_RAW_SHA256,
            },
            "control_dt_s": TRANSLATION_CONTROL_DT_S,
            "lead": {
                "hard_max_speed_m_s": (
                    TRANSLATION_LEAD_HARD_MAX_SPEED_M_S
                ),
                "hard_max_acceleration_m_s2": (
                    TRANSLATION_LEAD_HARD_MAX_ACCELERATION_M_S2
                ),
                "profile_acceleration_m_s2": (
                    TRANSLATION_LEAD_PROFILE_ACCELERATION_M_S2
                ),
                "progress_table_sha256": (
                    TRANSLATION_LEAD_PROGRESS_TABLE_SHA256
                ),
                "offsets_xy_sha256": (
                    TRANSLATION_LEAD_OFFSETS_XY_SHA256
                ),
                "kinematic_peaks_sha256": (
                    TRANSLATION_LEAD_KINEMATIC_PEAKS_SHA256
                ),
                "kinematic_peaks_by_shape": (
                    TRANSLATION_LEAD_KINEMATIC_PEAKS
                ),
            },
            "route": {
                "hard_max_speed_m_s": (
                    TRANSLATION_ROUTE_HARD_MAX_SPEED_M_S
                ),
                "hard_max_acceleration_m_s2": (
                    TRANSLATION_ROUTE_HARD_MAX_ACCELERATION_M_S2
                ),
                "profile_acceleration_m_s2": (
                    TRANSLATION_ROUTE_PROFILE_ACCELERATION_M_S2
                ),
                "stroke_ticks_by_shape": (
                    TRANSLATION_ROUTE_STROKE_TICKS
                ),
                "stroke_lengths_m_by_shape": (
                    tuple(
                        tuple(
                            _quantize_translation_schedule_value(value)
                            for value in row
                        )
                        for row in TRANSLATION_ROUTE_STROKE_LENGTHS_M
                    )
                ),
                "progress_table_sha256": (
                    TRANSLATION_ROUTE_PROGRESS_TABLE_SHA256
                ),
                "targets_xy_sha256": (
                    TRANSLATION_ROUTE_TARGETS_XY_SHA256
                ),
                "kinematic_peaks_sha256": (
                    TRANSLATION_ROUTE_KINEMATIC_PEAKS_SHA256
                ),
                "kinematic_peaks_by_shape": (
                    TRANSLATION_ROUTE_KINEMATIC_PEAKS
                ),
            },
        },
        "phase_invariant": True,
        "selection": "uniform_categorical_over_five_fixed_paths",
        "interpolation": "fixed_grid_minimum_jerk_c2_strokes",
        "lead_in": "activity_disk_center_to_first_path_point",
        "route_start_settle": "first_path_point_stationary_before_sampling",
        "terminal_hold": "final_path_point_stationary_after_sampling",
        "observation_anchors_are_timing_knots": False,
        "transforms": "none",
        "scale_shear_or_clip": False,
    }


def algorithm_contract_payload() -> dict[str, object]:
    """Return the canonical independent-tip training contract."""

    return {
        "schema": CONTRACT_SCHEMA,
        "training": {
            "algorithm": ALGORITHM,
            "physics_backend": PHYSICS_BACKEND,
            "torch_warp_exchange": TORCH_WARP_EXCHANGE,
            "solver": FORMAL_SOLVER,
            "constraint_jacobian": FORMAL_JACOBIAN,
            "xml_solver_tolerance": FORMAL_XML_SOLVER_TOLERANCE,
            "effective_solver_tolerance": (
                FORMAL_EFFECTIVE_SOLVER_TOLERANCE
            ),
            "dense_newton_max_nv": FORMAL_DENSE_NEWTON_MAX_NV,
            "physics_layout": PHYSICS_LAYOUT,
            "shard_object_order": SHARD_OBJECT_ORDER,
            "object_shard_count": OBJECT_SHARD_COUNT,
            "active_free_objects_per_shard": (
                ACTIVE_FREE_OBJECTS_PER_SHARD
            ),
            "cuda_graphs_per_global_batch": (
                CUDA_GRAPHS_PER_GLOBAL_BATCH
            ),
            "formal_global_worlds": FORMAL_GLOBAL_WORLDS,
            "formal_object_world_counts": FORMAL_OBJECT_WORLD_COUNTS,
            "formal_episode_ticks": FORMAL_EPISODE_TICKS,
            "formal_pre_action_ticks": FORMAL_PRE_ACTION_TICKS,
            "formal_remainder_worlds": FORMAL_REMAINDER_WORLDS,
            "formal_remainder_object_world_counts": (
                FORMAL_REMAINDER_OBJECT_WORLD_COUNTS
            ),
            "world_count_scope": "global_sum_not_per_shard",
            "rl_xml_excludes_untrained_plate": (
                RL_XML_EXCLUDES_UNTRAINED_PLATE
            ),
            "controllers": CONTROLLERS,
            "total_steps": TOTAL_TRAINING_STEPS,
            "step_definition": "mjwarp_closed_loop_ik_expert_transition",
            "training_flow": (
                "freeze_task_path_then_freeze_grasp_plane_and_four_anchors_"
                "then_certified_closed_loop_ik_demonstration_then_diffusion_"
                "imitation"
            ),
            "reward_guided_search": False,
            "diffusion_controls": "independent_hard_tip_ik_commands",
            "low_level_actuator_control": "per_tip_constrained_ik_kkt",
            "success_gate": SUCCESS_GATE_ENABLED,
            "expert_certification": {
                "stable_ticks": EXPERT_STABLE_TICKS,
                "prelift_contact_stable_ticks": (
                    PRELIFT_CONTACT_STABLE_TICKS
                ),
                "prelift_boundary_policy": (
                    "consecutive_exact_force_qualified_contacts_through_"
                    "the_lift_boundary"
                ),
                "translation_entry_policy": (
                    "consecutive_certificate_must_end_at_route_start"
                ),
                "minimum_hard_tip_object_contacts": (
                    EXPERT_MIN_CONTACTS
                ),
                "preferred_hard_tip_object_contacts": (
                    EXPERT_PREFERRED_CONTACTS
                ),
                "contact_identity": (
                    "native_contact_geom_pair_tip_and_active_object"
                ),
                "contact_direction_gate": False,
                "object_linear_speed_max_m_s": (
                    EXPERT_OBJECT_LINEAR_SPEED_MAX_M_S
                ),
                "object_angular_speed_max_rad_s": (
                    EXPERT_OBJECT_ANGULAR_SPEED_MAX_RAD_S
                ),
                "object_table_clearance_m": (
                    EXPERT_OBJECT_TABLE_CLEARANCE_M
                ),
                "tip_target_speed_max_m_s": TIP_TARGET_SPEED_MAX_M_S,
                "tip_target_acceleration_max_m_s2": (
                    TIP_TARGET_ACCELERATION_MAX_M_S2
                ),
                "tip_target_jerk_max_m_s3": TIP_TARGET_JERK_MAX_M_S3,
                "ik_cage_target_error_max_m": IK_CAGE_TARGET_ERROR_MAX_M,
                "ik_cage_projection": "shared_scale_across_four_hard_tips",
                "trust_projection_stages": (
                    "grasp",
                    "lift",
                    "path_start",
                    "route",
                ),
            },
            "object_drop": {
                "measurement": "body_xpos_z_plus_object_local_bottom_z",
                "table_top_z_m": TABLE_TOP_Z_M,
                "margin_m": DROP_MARGIN_M,
                "threshold_z_m": OBJECT_DROP_THRESHOLD_Z_M,
                "terminal_reward": OBJECT_DROP_TERMINAL_REWARD,
                "quarantine_position_m": OBJECT_DROP_QUARANTINE_POSITION_M,
                "continuation": "sticky_absorbing_quarantine_to_horizon",
                "nonfinite_policy": "independent_fail_fast",
            },
            "reset": {
                "active_object_com_z_m": INITIAL_OBJECT_COM_Z_M,
                "frame": "world",
                "support": "free_body_no_anchor",
            },
        },
        "objects": {
            "trained": OBJECTS,
            "classified_infeasible": tuple(sorted(INFEASIBLE_OBJECTS)),
            "rigid": tuple(sorted(RIGID_OBJECTS)),
            "fruit": tuple(sorted(FRUIT_OBJECTS)),
            "plate_trained": False,
        },
        "tasks": {
            "names": TASKS,
            "translation": translation_path_contract_payload(),
            "rotation": {
                "frame": ROTATION_FRAME,
                "axis_local": ROTATION_AXIS_LOCAL,
                "episode_goal": "closed_integer_angle_full_task",
                "minimum_angle_deg": ROTATION_ANGLE_MIN_DEG,
                "maximum_angle_deg": ROTATION_ANGLE_DEG,
                "motion_ticks": ROTATION_MOTION_TICKS,
                "terminal_hold_ticks": ROTATION_TERMINAL_HOLD_TICKS,
                "formal_action_ticks": ROTATION_FORMAL_ACTION_TICKS,
            },
        },
        "task_distribution": {
            "translation_path": {
                "phase_invariant": True,
                "fixed_shape_count": len(TRANSLATION_PATH_SHAPES),
                "equal_arc_intervals": TRANSLATION_PATH_SEGMENTS,
                "random_geometry": False,
            },
            "rotation_angle_integer_deg": (
                ROTATION_ANGLE_MIN_DEG,
                ROTATION_ANGLE_MAX_DEG,
            ),
            "phase_count": 1,
        },
        "reset_templates": {
            "count_per_object": RESET_TEMPLATES_PER_OBJECT,
            "seed_scheme": (
                "base + object_index * object_stride + template_index"
            ),
            "seed_base": RESET_TEMPLATE_SEED_BASE,
            "object_stride": RESET_TEMPLATE_OBJECT_STRIDE,
            "training_sampling": "uniform",
            "paired_evaluation": "unique_template_per_repeat",
        },
        "action": {
            "dim": ACTION_DIM,
            "tip_order": TIP_ORDER,
            "xyz_order": XYZ_ORDER,
            "normalized_limits": (
                NORMALIZED_ACTION_MIN,
                NORMALIZED_ACTION_MAX,
            ),
            "fields": tuple(
                {
                    "name": field.name,
                    "slice": (field.start, field.stop),
                    "shape": field.shape,
                    "scope": field.scope,
                    "applicability": field.applicability,
                    "physical_unit": field.physical_unit,
                    "physical_limits": (
                        field.physical_min,
                        field.physical_max,
                    ),
                }
                for field in ACTION_FIELDS
            ),
        },
        "success": {
            "translation": {
                "position_axis_abs_tolerance_m": (
                    TRANSLATION_POSITION_AXIS_ABS_TOLERANCE_M
                ),
                "euler_axis_abs_tolerance_deg": (
                    TRANSLATION_EULER_AXIS_ABS_TOLERANCE_DEG
                ),
            },
            "rotation": {
                "position_axis_abs_tolerance_m": (
                    ROTATION_POSITION_AXIS_ABS_TOLERANCE_M
                ),
                "euler_axis_abs_tolerance_deg": (
                    ROTATION_EULER_AXIS_ABS_TOLERANCE_DEG
                ),
            },
            "axis_order": XYZ_ORDER,
            "classification": "all_absolute_axes_within_closed_bounds",
            "l2_norms_are_diagnostic_only": True,
            "physical_hard_tip_plane_required": True,
            "anti_support_or_side_pinch_gate": False,
        },
        "grasp_plane": {
            "definition": "per_tip_commanded_z_in_actual_object_frame",
            "learned_objects": tuple(
                kind for kind in OBJECTS if grasp_plane_is_learned(kind)
            ),
            "learned_height_fraction_limits": (
                GRASP_PLANE_MIN_FRACTION,
                GRASP_PLANE_MAX_FRACTION,
            ),
            "fixed_nominal_square_height_fractions": tuple(
                (kind, grasp_plane_nominal_fraction(kind))
                for kind in OBJECTS
                if not grasp_plane_is_learned(kind)
            ),
            "fixed_nominal_yaw_rad": FIXED_GRASP_PLANE_YAW_RAD,
            "policy_residual_yaw_abs_limit_rad": YAW_LIMIT_RAD,
            "frozen_planner_yaw_principal_abs_limit_rad": (
                FROZEN_PLANNER_YAW_ABS_LIMIT_RAD
            ),
            "fixed_commanded_tip_z_exact": True,
            "fixed_physical_mean_z_tolerance_m": (
                FIXED_GRASP_PLANE_PHYSICAL_TOLERANCE_M
            ),
            "hard_projection": True,
            "commanded_and_actual_telemetry": True,
            "actual_measurement": "physical_hard_tip_body_positions",
            "projection_is_independent_per_tip": True,
            "instantaneous_residual_deformation_allowed": True,
            "independent_tip_xyz_residual_preserved": True,
            "fixed_square_scope": "nominal_contact_cage_only",
            "hard_tip_xy_coplanarity_required": False,
            "anti_support_gate": False,
        },
    }


def algorithm_signature() -> str:
    canonical = json.dumps(
        algorithm_contract_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


ALGORITHM_SIGNATURE = algorithm_signature()


def _translation_xy_table_respects_limits(
    table: tuple[tuple[tuple[float, float], ...], ...],
    maximum_speed_m_s: float,
    maximum_acceleration_m_s2: float,
) -> bool:
    """Check fixed-grid vector limits without regenerating canonical data."""

    dt = TRANSLATION_CONTROL_DT_S
    speed_limit_squared = (maximum_speed_m_s + 1.0e-12) ** 2
    acceleration_limit_squared = (
        maximum_acceleration_m_s2 + 1.0e-12
    ) ** 2
    for points in table:
        previous_velocity = (0.0, 0.0)
        for left, right in zip(points, points[1:]):
            velocity = (
                (right[0] - left[0]) / dt,
                (right[1] - left[1]) / dt,
            )
            acceleration = (
                (velocity[0] - previous_velocity[0]) / dt,
                (velocity[1] - previous_velocity[1]) / dt,
            )
            if (
                velocity[0] ** 2 + velocity[1] ** 2
                > speed_limit_squared
                or acceleration[0] ** 2 + acceleration[1] ** 2
                > acceleration_limit_squared
            ):
                return False
            previous_velocity = velocity
    return True


def validate_contract() -> None:
    if len(OBJECTS) != 4 or len(set(OBJECTS)) != len(OBJECTS):
        raise RuntimeError("the contract requires four unique training objects")
    if PLATE_OBJECT in OBJECTS:
        raise RuntimeError("plate must be excluded from training")
    if INFEASIBLE_OBJECTS != frozenset(("cylinder", "cone")):
        raise RuntimeError("the infeasible object classification changed")
    if INFEASIBLE_OBJECTS & frozenset(OBJECTS):
        raise RuntimeError("infeasible objects must be excluded from training")
    if RIGID_OBJECTS | FRUIT_OBJECTS != frozenset(OBJECTS):
        raise RuntimeError("rigid and fruit object sets must cover OBJECTS")
    if RIGID_OBJECTS & FRUIT_OBJECTS:
        raise RuntimeError("rigid and fruit object sets must be disjoint")
    if FORMAL_SOLVER != "Newton" or FORMAL_JACOBIAN != "dense":
        raise RuntimeError("formal MJWarp physics must use dense Newton")
    if not (
        0.0
        < FORMAL_XML_SOLVER_TOLERANCE
        < FORMAL_EFFECTIVE_SOLVER_TOLERANCE
    ):
        raise RuntimeError(
            "formal XML tolerance must be below the MJWarp float32 floor"
        )
    if FORMAL_DENSE_NEWTON_MAX_NV != 60:
        raise RuntimeError("formal dense Newton nv limit must remain 60")
    if not RL_XML_EXCLUDES_UNTRAINED_PLATE:
        raise RuntimeError("formal RL XML must exclude the untrained plate")
    if (
        PHYSICS_LAYOUT != "task_specific_four_single_object_xml_shards"
        or SHARD_OBJECT_ORDER != OBJECTS
        or OBJECT_SHARD_COUNT != 4
        or ACTIVE_FREE_OBJECTS_PER_SHARD != 1
        or CUDA_GRAPHS_PER_GLOBAL_BATCH != 4
    ):
        raise RuntimeError("formal physics must use four ordered object shards")
    if (
        FORMAL_GLOBAL_WORLDS != 20_000
        or len(FORMAL_OBJECT_WORLD_COUNTS) != OBJECT_SHARD_COUNT
        or FORMAL_OBJECT_WORLD_COUNTS != (5_000,) * OBJECT_SHARD_COUNT
        or sum(FORMAL_OBJECT_WORLD_COUNTS) != FORMAL_GLOBAL_WORLDS
        or any(worlds % 10 for worlds in FORMAL_OBJECT_WORLD_COUNTS)
    ):
        raise RuntimeError("formal object-shard world allocation is invalid")
    if (
        FORMAL_REMAINDER_WORLDS != 0
        or sum(FORMAL_REMAINDER_OBJECT_WORLD_COUNTS)
        != FORMAL_REMAINDER_WORLDS
        or any(
            worlds % 10
            for worlds in FORMAL_REMAINDER_OBJECT_WORLD_COUNTS
        )
    ):
        raise RuntimeError("formal phase remainder allocation is invalid")
    if (
        FORMAL_REMAINDER_OBJECT_WORLD_COUNTS != (0,) * 4
    ):
        raise RuntimeError("formal run must not use remainder shards")
    if not (
        GRASP_PLANE_MIN_FRACTION == 0.40
        and GRASP_PLANE_NOMINAL_FRACTION == 0.50
        and GRASP_PLANE_MAX_FRACTION == 0.60
    ):
        raise RuntimeError("grasp-plane fractions must be ordered inside (0, 1)")
    if not (
        ROTATION_ANGLE_MIN_DEG == 10
        and ROTATION_ANGLE_MIN_DEG
        == ROTATION_EULER_AXIS_ABS_TOLERANCE_DEG
        and ROTATION_ANGLE_MAX_DEG == 45
        and ROTATION_ANGLE_MIN_DEG <= ROTATION_ANGLE_MAX_DEG
        == ROTATION_ANGLE_DEG
    ):
        raise RuntimeError("rotation curriculum intervals are invalid")
    if not (
        TRANSLATION_PATH_SEGMENTS == 20
        and TRANSLATION_PATH_WAYPOINTS == 21
        and len(TRANSLATION_PATH_SHAPES) == 5
        and len(TRANSLATION_PATH_LENGTH_M_BY_SHAPE) == 5
        and len(TRANSLATION_PATH_STEP_M_BY_SHAPE) == 5
        and len(TRANSLATION_PATH_WAYPOINTS_XY_M) == 5
        and all(
            len(points) == TRANSLATION_PATH_WAYPOINTS
            for points in TRANSLATION_PATH_WAYPOINTS_XY_M
        )
        and 0.0 < TRANSLATION_PATH_START_RADIUS_MAX_M
        and TRANSLATION_PATH_START_RADIUS_MAX_M
        <= TRANSLATION_PATH_REGION_RADIUS_M
        and TRANSLATION_PATH_GENERATION_RADIUS_M
        == TRANSLATION_PATH_REGION_RADIUS_M
        and all(
            math.hypot(*point) <= TRANSLATION_PATH_REGION_RADIUS_M + 1.0e-12
            for points in TRANSLATION_PATH_WAYPOINTS_XY_M
            for point in points
        )
        and all(
            math.isclose(
                step * TRANSLATION_PATH_SEGMENTS,
                length,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for step, length in zip(
                TRANSLATION_PATH_STEP_M_BY_SHAPE,
                TRANSLATION_PATH_LENGTH_M_BY_SHAPE,
            )
        )
        and math.isclose(
            TRANSLATION_PATH_LENGTH_M_BY_SHAPE[0], 0.400,
            rel_tol=0.0, abs_tol=1.0e-12,
        )
        and math.isclose(
            TRANSLATION_PATH_LENGTH_M_BY_SHAPE[1], 0.100 * math.pi,
            rel_tol=0.0, abs_tol=1.0e-12,
        )
        and all(
            math.isclose(length, 0.200, rel_tol=0.0, abs_tol=1.0e-12)
            for length in TRANSLATION_PATH_LENGTH_M_BY_SHAPE[2:]
        )
        and all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
            for left, right in zip(
                TRANSLATION_PATH_WAYPOINTS_XY_M[0][0],
                TRANSLATION_PATH_WAYPOINTS_XY_M[0][-1],
            )
        )
        and all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
            for left, right in zip(
                TRANSLATION_PATH_WAYPOINTS_XY_M[1][0],
                TRANSLATION_PATH_WAYPOINTS_XY_M[1][-1],
            )
        )
        and all(
            math.isclose(math.hypot(*point), TRANSLATION_CIRCLE_RADIUS_M,
                         rel_tol=0.0, abs_tol=1.0e-12)
            for point in TRANSLATION_PATH_WAYPOINTS_XY_M[1]
        )
        and TRANSLATION_PATH_SEGMENT_TICKS == 90
        and TRANSLATION_PATH_LEAD_IN_TICKS
        + TRANSLATION_PATH_START_SETTLE_TICKS
        + TRANSLATION_PATH_ROUTE_SAMPLES
        + TRANSLATION_PATH_TERMINAL_HOLD_TICKS
        == FORMAL_ACTION_TICKS
        and TRANSLATION_PATH_LEAD_IN_TICKS == 300
        and TRANSLATION_PATH_ROUTE_SAMPLES == 1_800
        and TRANSLATION_PATH_START_SETTLE_TICKS == 300
        and TRANSLATION_PATH_TERMINAL_HOLD_TICKS == 1_802
        and TRANSLATION_PATH_ROUTE_SAMPLES
        == TRANSLATION_PATH_SEGMENTS
        * TRANSLATION_PATH_SEGMENT_TICKS
        and ROTATION_MOTION_TICKS + ROTATION_TERMINAL_HOLD_TICKS
        == ROTATION_FORMAL_ACTION_TICKS
        and FORMAL_PRE_ACTION_TICKS + FORMAL_ACTION_TICKS
        == FORMAL_EPISODE_TICKS
        and FORMAL_PRE_ACTION_TICKS + ROTATION_FORMAL_ACTION_TICKS
        == FORMAL_EPISODE_TICKS
    ):
        raise RuntimeError("translation path geometry/timing contract is invalid")
    schedule_rows_valid = all(
        len(route) == TRANSLATION_PATH_ROUTE_SAMPLES + 1
        and len(lead) == TRANSLATION_PATH_LEAD_IN_TICKS + 1
        and route[0] == 0.0
        and route[-1] == 1.0
        and lead[0] == 0.0
        and lead[-1] == 1.0
        and all(math.isfinite(value) for value in (*route, *lead))
        and all(
            value == _quantize_translation_schedule_value(value)
            for value in (*route, *lead)
        )
        and all(left <= right for left, right in zip(route, route[1:]))
        and all(left <= right for left, right in zip(lead, lead[1:]))
        for route, lead in zip(
            TRANSLATION_ROUTE_PROGRESS_TABLE,
            TRANSLATION_LEAD_PROGRESS_TABLE,
        )
    )
    canonical_xy_valid = (
        len(TRANSLATION_PATH_WAYPOINTS_XY_M)
        == len(TRANSLATION_PATH_SHAPES)
        and len(TRANSLATION_ROUTE_TARGETS_XY_M)
        == len(TRANSLATION_PATH_SHAPES)
        and len(TRANSLATION_LEAD_OFFSETS_XY_M)
        == len(TRANSLATION_PATH_SHAPES)
        and all(
            len(anchors) == TRANSLATION_PATH_WAYPOINTS
            and len(route) == TRANSLATION_PATH_ROUTE_SAMPLES + 1
            and len(lead) == TRANSLATION_PATH_LEAD_IN_TICKS + 1
            and route[0] == anchors[0]
            and route[-1] == anchors[-1]
            and lead[0] == (0.0, 0.0)
            and lead[-1] == anchors[0]
            and all(
                math.isfinite(value)
                and value == _quantize_translation_schedule_value(value)
                for rows in (anchors, route, lead)
                for point in rows
                for value in point
            )
            for anchors, route, lead in zip(
                TRANSLATION_PATH_WAYPOINTS_XY_M,
                TRANSLATION_ROUTE_TARGETS_XY_M,
                TRANSLATION_LEAD_OFFSETS_XY_M,
            )
        )
    )
    stroke_contract_valid = all(
        sum(ticks) == TRANSLATION_PATH_ROUTE_SAMPLES
        and math.isclose(
            sum(lengths),
            total_length,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and len(ticks) == len(lengths)
        and all(value > 0 for value in ticks)
        and all(value > 0.0 for value in lengths)
        for ticks, lengths, total_length in zip(
            TRANSLATION_ROUTE_STROKE_TICKS,
            TRANSLATION_ROUTE_STROKE_LENGTHS_M,
            TRANSLATION_PATH_LENGTH_M_BY_SHAPE,
        )
    )
    schedule_peaks_valid = (
        len(TRANSLATION_ROUTE_KINEMATIC_PEAKS)
        == len(TRANSLATION_PATH_SHAPES)
        and len(TRANSLATION_LEAD_KINEMATIC_PEAKS)
        == len(TRANSLATION_PATH_SHAPES)
        and all(
            speed <= TRANSLATION_ROUTE_HARD_MAX_SPEED_M_S + 1.0e-12
            and acceleration
            <= TRANSLATION_ROUTE_HARD_MAX_ACCELERATION_M_S2 + 1.0e-12
            for speed, acceleration in TRANSLATION_ROUTE_KINEMATIC_PEAKS
        )
        and all(
            speed <= TRANSLATION_LEAD_HARD_MAX_SPEED_M_S + 1.0e-12
            and acceleration
            <= TRANSLATION_LEAD_HARD_MAX_ACCELERATION_M_S2 + 1.0e-12
            for speed, acceleration in TRANSLATION_LEAD_KINEMATIC_PEAKS
        )
        and all(
            value == _quantize_translation_schedule_value(value)
            for peaks in (
                *TRANSLATION_ROUTE_KINEMATIC_PEAKS,
                *TRANSLATION_LEAD_KINEMATIC_PEAKS,
            )
            for value in peaks
        )
    )
    if not (
        len(TRANSLATION_ROUTE_PROGRESS_TABLE)
        == len(TRANSLATION_PATH_SHAPES)
        and len(TRANSLATION_LEAD_PROGRESS_TABLE)
        == len(TRANSLATION_PATH_SHAPES)
        and schedule_rows_valid
        and canonical_xy_valid
        and stroke_contract_valid
        and schedule_peaks_valid
        and _translation_xy_table_respects_limits(
            TRANSLATION_ROUTE_TARGETS_XY_M,
            TRANSLATION_ROUTE_HARD_MAX_SPEED_M_S,
            TRANSLATION_ROUTE_HARD_MAX_ACCELERATION_M_S2,
        )
        and _translation_xy_table_respects_limits(
            TRANSLATION_LEAD_OFFSETS_XY_M,
            TRANSLATION_LEAD_HARD_MAX_SPEED_M_S,
            TRANSLATION_LEAD_HARD_MAX_ACCELERATION_M_S2,
        )
        and TRANSLATION_SCHEDULE_QUANTIZATION_DECIMALS == 12
        and TRANSLATION_SCHEDULE_QUANTIZATION_SCALE == 1_000_000_000_000
        and TRANSLATION_SCHEDULE_TABLE_HASH_ENCODING
        == "sha256_big_endian_signed_int64_grid_1e-12"
        and TRANSLATION_CANONICAL_BLOB_ENCODING
        == "base85_zlib_be_i64_delta_v1"
        and TRANSLATION_CANONICAL_BLOB_VERSION == 1
        and TRANSLATION_CONTROL_DT_S == 0.020
        and 0.0
        < TRANSLATION_ROUTE_PROFILE_ACCELERATION_M_S2
        < TRANSLATION_ROUTE_HARD_MAX_ACCELERATION_M_S2
        and 0.0
        < TRANSLATION_LEAD_PROFILE_ACCELERATION_M_S2
        < TRANSLATION_LEAD_HARD_MAX_ACCELERATION_M_S2
        and 0.0 < TRANSLATION_ROUTE_HARD_MAX_SPEED_M_S
        and 0.0 < TRANSLATION_LEAD_HARD_MAX_SPEED_M_S
        and len(TRANSLATION_ROUTE_PROGRESS_TABLE_SHA256) == 64
        and len(TRANSLATION_LEAD_PROGRESS_TABLE_SHA256) == 64
        and all(
            len(value) == 64
            for value in (
                TRANSLATION_CANONICAL_BLOB_SHA256,
                TRANSLATION_CANONICAL_COMPRESSED_SHA256,
                TRANSLATION_CANONICAL_RAW_SHA256,
                TRANSLATION_PATH_WAYPOINTS_XY_SHA256,
                TRANSLATION_ROUTE_TARGETS_XY_SHA256,
                TRANSLATION_LEAD_OFFSETS_XY_SHA256,
                TRANSLATION_ROUTE_KINEMATIC_PEAKS_SHA256,
                TRANSLATION_LEAD_KINEMATIC_PEAKS_SHA256,
            )
        )
    ):
        raise RuntimeError("translation bounded schedule contract is invalid")
    if not (
        TRANSLATION_POSITION_AXIS_ABS_TOLERANCE_M == 0.010
        and TRANSLATION_EULER_AXIS_ABS_TOLERANCE_DEG == 20.0
        and TRANSLATION_PATH_TRACKING_AXIS_ABS_TOLERANCE_M == 0.010
        and ROTATION_POSITION_AXIS_ABS_TOLERANCE_M == 0.010
        and ROTATION_EULER_AXIS_ABS_TOLERANCE_DEG == 10.0
    ):
        raise RuntimeError("task-dependent endpoint/path tolerances are invalid")
    if not (
        EXPERT_STABLE_TICKS == 25
        and PRELIFT_CONTACT_STABLE_TICKS == 5
        and EXPERT_MIN_CONTACTS == 3
        and EXPERT_PREFERRED_CONTACTS == len(TIP_ORDER)
        and 0.0 < EXPERT_OBJECT_LINEAR_SPEED_MAX_M_S <= 0.005
        and 0.0 < EXPERT_OBJECT_ANGULAR_SPEED_MAX_RAD_S <= 0.05
        and 0.0 < EXPERT_OBJECT_TABLE_CLEARANCE_M < 0.01
        and 0.0 < TIP_TARGET_SPEED_MAX_M_S
        and 0.0 < TIP_TARGET_ACCELERATION_MAX_M_S2
        and 0.0 < TIP_TARGET_JERK_MAX_M_S3
        and IK_CAGE_TARGET_ERROR_MAX_M == 0.003
    ):
        raise RuntimeError("expert physical certification contract changed")
    if (
        FIXED_GRASP_PLANE_FRACTIONS
        or LEARNED_GRASP_PLANE_OBJECTS != FRUIT_OBJECTS
        or any(
            not 0.0 < float(fraction) < 1.0
            for fraction in FIXED_GRASP_PLANE_FRACTIONS.values()
        )
        or FIXED_GRASP_PLANE_YAW_RAD != 0.0
        or FROZEN_PLANNER_YAW_ABS_LIMIT_RAD != math.pi
        or FROZEN_PLANNER_YAW_ABS_LIMIT_RAD <= YAW_LIMIT_RAD
        or not 0.0 < FIXED_GRASP_PLANE_PHYSICAL_TOLERANCE_M < 0.01
    ):
        raise RuntimeError("fixed and learned grasp-plane contracts are invalid")
    if any(
        grasp_plane_fraction_limits(kind)
        != (
            (GRASP_PLANE_MIN_FRACTION, GRASP_PLANE_MAX_FRACTION)
            if kind in FRUIT_OBJECTS
            else (grasp_plane_nominal_fraction(kind),) * 2
        )
        for kind in OBJECTS
    ):
        raise RuntimeError("object-specific grasp-plane limits are inconsistent")
    if not (
        0.0 < DROP_MARGIN_M < TABLE_TOP_Z_M
        and INITIAL_OBJECT_COM_Z_M > TABLE_TOP_Z_M
        and math.isfinite(INITIAL_OBJECT_COM_Z_M)
        and math.isclose(
            OBJECT_DROP_THRESHOLD_Z_M,
            TABLE_TOP_Z_M - DROP_MARGIN_M,
        )
        and OBJECT_DROP_TERMINAL_REWARD == -10.0
        and len(OBJECT_DROP_QUARANTINE_POSITION_M) == 3
        and all(
            math.isfinite(value)
            for value in OBJECT_DROP_QUARANTINE_POSITION_M
        )
        and math.hypot(*OBJECT_DROP_QUARANTINE_POSITION_M[:2]) > 1.0
    ):
        raise RuntimeError("object-drop termination constants are invalid")
    covered = tuple(
        index
        for field in ACTION_FIELDS
        for index in range(field.start, field.stop)
    )
    if covered != tuple(range(ACTION_DIM)):
        raise RuntimeError("action fields must partition the 25-D action")
    if set(EPISODE_LEVEL_ACTION_INDICES) & set(DYNAMIC_ACTION_INDICES):
        raise RuntimeError("episode and dynamic action channels overlap")
    if sorted(EPISODE_LEVEL_ACTION_INDICES + DYNAMIC_ACTION_INDICES) != list(
        range(ACTION_DIM)
    ):
        raise RuntimeError("episode and dynamic channels must cover the action")
    if RESET_TEMPLATES_PER_OBJECT <= 0:
        raise RuntimeError("the reset template bank must not be empty")
    reset_seeds = {
        reset_template_seed(object_index, template_index)
        for object_index in range(len(OBJECTS))
        for template_index in range(RESET_TEMPLATES_PER_OBJECT)
    }
    if len(reset_seeds) != len(OBJECTS) * RESET_TEMPLATES_PER_OBJECT:
        raise RuntimeError("reset template seed scheme contains collisions")
    if decode_action(ACTION_LOW, "banana").force_scale != (
        FORCE_SCALE_MIN,
    ) * len(TIP_ORDER):
        raise RuntimeError("force lower endpoint mapping is inconsistent")
    if decode_action(ACTION_HIGH, "banana").force_scale != (
        FORCE_SCALE_MAX,
    ) * len(TIP_ORDER):
        raise RuntimeError("force upper endpoint mapping is inconsistent")
    if decode_action(ACTION_LOW, "banana").tip_height_fraction != (
        GRASP_PLANE_MIN_FRACTION,
    ) * len(TIP_ORDER):
        raise RuntimeError("fruit height lower endpoint mapping is inconsistent")
    if decode_action(ACTION_HIGH, "banana").tip_height_fraction != (
        GRASP_PLANE_MAX_FRACTION,
    ) * len(TIP_ORDER):
        raise RuntimeError("fruit height upper endpoint mapping is inconsistent")
    for kind, fraction in FIXED_GRASP_PLANE_FRACTIONS.items():
        decoded = decode_action(ACTION_HIGH, kind)
        if decoded.tip_height_fraction != (fraction,) * len(TIP_ORDER):
            raise RuntimeError("fixed height endpoint mapping is inconsistent")
        if decoded.yaw_rad != FIXED_GRASP_PLANE_YAW_RAD:
            raise RuntimeError("fixed yaw endpoint mapping is inconsistent")


validate_contract()


__all__ = [
    "ACTION_DIM",
    "ACTION_REFERENCE_CAPTURE",
    "ACTIVE_FREE_OBJECTS_PER_SHARD",
    "ACTION_FIELDS",
    "ACTION_HIGH",
    "ACTION_LIMITS",
    "ACTION_LOW",
    "ACTION_SLICES",
    "ALGORITHM",
    "ALGORITHM_SIGNATURE",
    "CONTROLLERS",
    "CUDA_GRAPHS_PER_GLOBAL_BATCH",
    "DYNAMIC_ACTION_INDICES",
    "DYNAMIC_ACTION_SLICE",
    "DROP_MARGIN_M",
    "DecodedAction",
    "EXPERT_MIN_CONTACTS",
    "EXPERT_OBJECT_ANGULAR_SPEED_MAX_RAD_S",
    "EXPERT_OBJECT_LINEAR_SPEED_MAX_M_S",
    "EXPERT_OBJECT_TABLE_CLEARANCE_M",
    "EXPERT_PREFERRED_CONTACTS",
    "EXPERT_STABLE_TICKS",
    "PRELIFT_CONTACT_STABLE_TICKS",
    "EPISODE_LEVEL_ACTION_INDICES",
    "EPISODE_LEVEL_ACTION_SLICES",
    "FORCE_SCALE_INDICES",
    "FORCE_SCALE_MAX",
    "FORCE_SCALE_MIN",
    "FORCE_SCALE_NOMINAL",
    "FORMAL_DENSE_NEWTON_MAX_NV",
    "FORMAL_ACTION_TICKS",
    "FORMAL_EPISODE_TICKS",
    "FORMAL_EFFECTIVE_SOLVER_TOLERANCE",
    "FORMAL_GLOBAL_WORLDS",
    "FORMAL_JACOBIAN",
    "FORMAL_SOLVER",
    "FORMAL_OBJECT_WORLD_COUNTS",
    "FORMAL_PRE_ACTION_TICKS",
    "FORMAL_REMAINDER_WORLDS",
    "FORMAL_REMAINDER_OBJECT_WORLD_COUNTS",
    "FORMAL_PHASE_REMAINDER_WORLDS",
    "FORMAL_PHASE1_REMAINDER_OBJECT_WORLD_COUNTS",
    "FORMAL_PHASE2_REMAINDER_OBJECT_WORLD_COUNTS",
    "FORMAL_XML_SOLVER_TOLERANCE",
    "FIXED_GRASP_PLANE_FRACTIONS",
    "FIXED_GRASP_PLANE_PHYSICAL_TOLERANCE_M",
    "FIXED_GRASP_PLANE_YAW_RAD",
    "FRUIT_OBJECTS",
    "GRASP_PLANE_MAX_FRACTION",
    "GRASP_PLANE_MIN_FRACTION",
    "GRASP_PLANE_NOMINAL_FRACTION",
    "INFEASIBLE_OBJECTS",
    "IK_CAGE_TARGET_ERROR_MAX_M",
    "INITIAL_OBJECT_COM_Z_M",
    "LEARNED_GRASP_PLANE_OBJECTS",
    "OBJECTS",
    "OBJECT_DROP_TERMINAL_REWARD",
    "OBJECT_DROP_QUARANTINE_POSITION_M",
    "OBJECT_DROP_THRESHOLD_Z_M",
    "OBJECT_SHARD_COUNT",
    "PLATE_OBJECT",
    "PHYSICS_BACKEND",
    "PHYSICS_LAYOUT",
    "PHASE1_ROTATION_ANGLE_MAX_DEG",
    "PHASE1_ROTATION_ANGLE_MIN_DEG",
    "PHASE1_TRANSLATION_PATH_SEGMENT_TICKS",
    "PHASE2_ROTATION_ANGLE_MAX_DEG",
    "PHASE2_ROTATION_ANGLE_MIN_DEG",
    "PHASE2_TRANSLATION_PATH_SEGMENT_TICKS",
    "RIGID_OBJECTS",
    "ROTATION_ANGLE_DEG",
    "ROTATION_ANGLE_MAX_DEG",
    "ROTATION_ANGLE_MIN_DEG",
    "ROTATION_AXIS_LOCAL",
    "ROTATION_FRAME",
    "ROTATION_EULER_AXIS_ABS_TOLERANCE_DEG",
    "ROTATION_FORMAL_ACTION_TICKS",
    "ROTATION_MOTION_TICKS",
    "ROTATION_TERMINAL_HOLD_TICKS",
    "ROTATION_POSITION_AXIS_ABS_TOLERANCE_M",
    "RESET_TEMPLATES_PER_OBJECT",
    "RESET_TEMPLATE_OBJECT_STRIDE",
    "RESET_TEMPLATE_SEED_BASE",
    "RL_XML_EXCLUDES_UNTRAINED_PLATE",
    "SUCCESS_GATE_ENABLED",
    "SHARD_OBJECT_ORDER",
    "TASK_ROTATE",
    "TASK_TRANSLATE",
    "TASKS",
    "TABLE_TOP_Z_M",
    "TIP_HEIGHT_SLICE",
    "TIP_ORDER",
    "TIP_TARGET_ACCELERATION_MAX_M_S2",
    "TIP_TARGET_JERK_MAX_M_S3",
    "TIP_TARGET_SPEED_MAX_M_S",
    "TIP_PRELOAD_LIMIT_M",
    "TIP_PRELOAD_SLICE",
    "TIP_XYZ_RESIDUAL_LIMIT_M",
    "TIP_XYZ_RESIDUAL_SLICE",
    "TOTAL_TRAINING_STEPS",
    "TORCH_WARP_EXCHANGE",
    "TRANSLATION_DZ_M",
    "TRANSLATION_FRAME",
    "TRANSLATION_EULER_AXIS_ABS_TOLERANCE_DEG",
    "TRANSLATION_PATH_TRACKING_AXIS_ABS_TOLERANCE_M",
    "TRANSLATION_PATH_FAMILY",
    "TRANSLATION_PATH_LENGTH_M_BY_SHAPE",
    "TRANSLATION_PATH_LEAD_IN_TICKS",
    "TRANSLATION_PATH_START_SETTLE_TICKS",
    "TRANSLATION_PATH_TERMINAL_HOLD_TICKS",
    "TRANSLATION_PATH_ROUTE_SAMPLES",
    "TRANSLATION_PATH_MAX_LENGTH_M",
    "TRANSLATION_PATH_MAX_STEP_M",
    "TRANSLATION_PATH_REGION_RADIUS_M",
    "TRANSLATION_PATH_SEED_MAX",
    "TRANSLATION_PATH_SEED_MIN",
    "TRANSLATION_PATH_SEGMENTS",
    "TRANSLATION_PATH_SEGMENT_TICKS",
    "TRANSLATION_PATH_SHAPES",
    "TRANSLATION_PATH_STEP_M_BY_SHAPE",
    "TRANSLATION_PATH_WAYPOINTS",
    "TRANSLATION_PATH_WAYPOINTS_XY_M",
    "TRANSLATION_PATH_WAYPOINTS_XY_SHA256",
    "TRANSLATION_POSITION_AXIS_ABS_TOLERANCE_M",
    "TRANSLATION_PATH_GENERATION_RADIUS_M",
    "TRANSLATION_PATH_RELATIVE_EXTENT_MAX_M",
    "TRANSLATION_PATH_START_RADIUS_MAX_M",
    "TRANSLATION_CONTROL_DT_S",
    "TRANSLATION_SCHEDULE_QUANTIZATION_DECIMALS",
    "TRANSLATION_SCHEDULE_QUANTIZATION_SCALE",
    "TRANSLATION_SCHEDULE_TABLE_HASH_ENCODING",
    "TRANSLATION_CANONICAL_BLOB_ENCODING",
    "TRANSLATION_CANONICAL_BLOB_VERSION",
    "TRANSLATION_CANONICAL_BLOB_SHA256",
    "TRANSLATION_CANONICAL_COMPRESSED_SHA256",
    "TRANSLATION_CANONICAL_RAW_SHA256",
    "TRANSLATION_LEAD_HARD_MAX_SPEED_M_S",
    "TRANSLATION_LEAD_HARD_MAX_ACCELERATION_M_S2",
    "TRANSLATION_LEAD_PROFILE_ACCELERATION_M_S2",
    "TRANSLATION_LEAD_PROGRESS_TABLE",
    "TRANSLATION_LEAD_PROGRESS_TABLE_SHA256",
    "TRANSLATION_LEAD_OFFSETS_XY_M",
    "TRANSLATION_LEAD_OFFSETS_XY_SHA256",
    "TRANSLATION_LEAD_KINEMATIC_PEAKS",
    "TRANSLATION_LEAD_KINEMATIC_PEAKS_SHA256",
    "TRANSLATION_ROUTE_HARD_MAX_SPEED_M_S",
    "TRANSLATION_ROUTE_HARD_MAX_ACCELERATION_M_S2",
    "TRANSLATION_ROUTE_PROFILE_ACCELERATION_M_S2",
    "TRANSLATION_ROUTE_STROKE_TICKS",
    "TRANSLATION_ROUTE_STROKE_LENGTHS_M",
    "TRANSLATION_ROUTE_PROGRESS_TABLE",
    "TRANSLATION_ROUTE_PROGRESS_TABLE_SHA256",
    "TRANSLATION_ROUTE_TARGETS_XY_M",
    "TRANSLATION_ROUTE_TARGETS_XY_SHA256",
    "TRANSLATION_ROUTE_KINEMATIC_PEAKS",
    "TRANSLATION_ROUTE_KINEMATIC_PEAKS_SHA256",
    "TRANSLATION_CIRCLE_RADIUS_M",
    "TRANSLATION_LETTER_PATH_LENGTH_M",
    "TRANSLATION_SQUARE_SIDE_M",
    "TRANSLATION_U_RADIUS_M",
    "YAW_INDEX",
    "FROZEN_PLANNER_YAW_ABS_LIMIT_RAD",
    "YAW_LIMIT_DEG",
    "YAW_LIMIT_RAD",
    "YAW_SLICE",
    "ZERO_ACTION",
    "algorithm_contract_payload",
    "algorithm_signature",
    "decode_action",
    "dynamic_action",
    "episode_level_action",
    "grasp_plane_fraction_limits",
    "grasp_plane_is_learned",
    "grasp_plane_nominal_fraction",
    "mask_action_for_object",
    "reset_template_seed",
    "translation_path_contract_payload",
    "translation_path_point_xy_m",
    "validate_action",
    "validate_contract",
    "validate_episode_action_stability",
    "validate_object",
    "validate_task",
]
