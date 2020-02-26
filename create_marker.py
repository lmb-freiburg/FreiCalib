# Christian Zimmermann, 2019
# Adapted from Kalibr: https://github.com/ethz-asl/kalibr
# Thomas Schneider, Sept 2013
# Codes from AprilTags C++ Library (http://people.csail.mit.edu/kaess/apriltags/)

"""
FROM THE ORIGINAL LICENSE FILE

Copyright (c) 2014, Paul Furgale, Jérôme Maye and Jörn Rehder, Autonomous Systems Lab,
                    ETH Zurich, Switzerland
Copyright (c) 2014, Thomas Schneider, Skybotix AG, Switzerland
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    All advertising materials mentioning features or use of this software must
    display the following acknowledgement: This product includes software developed
    by the Autonomous Systems Lab and Skybotix AG.

    Neither the name of the Autonomous Systems Lab and Skybotix AG nor the names
    of its contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTONOMOUS SYSTEMS LAB AND SKYBOTIX AG ''AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL the AUTONOMOUS SYSTEMS LAB OR SKYBOTIX AG BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
"""

from pyx import *
import argparse
import sys

import os
import math
import numpy as np

from utils.general_util import json_dump, my_mkdir


class AprilTagCodes:
    t16h5=[0x231b, 0x2ea5, 0x346a, 0x45b9, 0x79a6, 0x7f6b, 0xb358, 0xe745, 0xfe59, 0x156d, 0x380b, 0xf0ab, 0x0d84, 0x4736, 0x8c72, 0xaf10, 0x093c, 0x93b4, 0xa503, 0x468f, 0xe137, 0x5795, 0xdf42, 0x1c1d, 0xe9dc, 0x73ad, 0xad5f, 0xd530, 0x07ca, 0xaf2e]
    t25h7=[0x4b770d, 0x11693e6, 0x1a599ab, 0xc3a535, 0x152aafa, 0xaccd98, 0x1cad922, 0x2c2fad, 0xbb3572, 0x14a3b37, 0x186524b, 0xc99d4c, 0x23bfea, 0x141cb74, 0x1d0d139, 0x1670aeb, 0x851675, 0x150334e, 0x6e3ed8, 0xfd449d, 0xaa55ec, 0x1c86176, 0x15e9b28, 0x7ca6b2, 0x147c38b, 0x1d6c950, 0x8b0e8c, 0x11a1451, 0x1562b65, 0x13f53c8, 0xd58d7a, 0x829ec9, 0xfaccf1, 0x136e405, 0x7a2f06, 0x10934cb, 0x16a8b56, 0x1a6a26a, 0xf85545, 0x195c2e4, 0x24c8a9, 0x12bfc96, 0x16813aa, 0x1a42abe, 0x1573424, 0x1044573, 0xb156c2, 0x5e6811, 0x1659bfe, 0x1d55a63, 0x5bf065, 0xe28667, 0x1e9ba54, 0x17d7c5a, 0x1f5aa82, 0x1a2bbd1, 0x1ae9f9, 0x1259e51, 0x134062b, 0xe1177a, 0xed07a8, 0x162be24, 0x59128b, 0x1663e8f, 0x1a83cb, 0x45bb59, 0x189065a, 0x4bb370, 0x16fb711, 0x122c077, 0xeca17a, 0xdbc1f4, 0x88d343, 0x58ac5d, 0xba02e8, 0x1a1d9d, 0x1c72eec, 0x924bc5, 0xdccab3, 0x886d15, 0x178c965, 0x5bc69a, 0x1716261, 0x174e2cc, 0x1ed10f4, 0x156aa8, 0x3e2a8a, 0x2752ed, 0x153c651, 0x1741670, 0x765b05, 0x119c0bb, 0x172a783, 0x4faca1, 0xf31257, 0x12441fc, 0x0d3748, 0xc21f15, 0xac5037, 0x180e592, 0x7d3210, 0xa27187, 0x2beeaf, 0x26ff57, 0x690e82, 0x77765c, 0x1a9e1d7, 0x140be1a, 0x1aa1e3a, 0x1944f5c, 0x19b5032, 0x169897, 0x1068eb9, 0xf30dbc, 0x106a151, 0x1d53e95, 0x1348cee, 0xcf4fca, 0x1728bb5, 0xdc1eec, 0x69e8db, 0x16e1523, 0x105fa25, 0x18abb0c, 0xc4275d, 0x6d8e76, 0xe8d6db, 0xe16fd7, 0x1ac2682, 0x77435b, 0xa359dd, 0x3a9c4e, 0x123919a, 0x1e25817, 0x02a836, 0x1545a4, 0x1209c8d, 0xbb5f69, 0x1dc1f02, 0x5d5f7e, 0x12d0581, 0x13786c2, 0xe15409, 0x1aa3599, 0x139aad8, 0xb09d2a, 0x54488f, 0x13c351c, 0x976079, 0xb25b12, 0x1addb34, 0x1cb23ae, 0x1175738, 0x1303bb8, 0xd47716, 0x188ceea, 0xbaf967, 0x1226d39, 0x135e99b, 0x34adc5, 0x2e384d, 0x90d3fa, 0x232713, 0x17d49b1, 0xaa84d6, 0xc2ddf8, 0x1665646, 0x4f345f, 0x2276b1, 0x1255dd7, 0x16f4ccc, 0x4aaffc, 0xc46da6, 0x85c7b3, 0x1311fcb, 0x9c6c4f, 0x187d947, 0x8578e4, 0xe2bf0b, 0xa01b4c, 0xa1493b, 0x7ad766, 0xccfe82, 0x1981b5b, 0x1cacc85, 0x562cdb, 0x15b0e78, 0x8f66c5, 0x3332bf, 0x12ce754, 0x096a76, 0x1d5e3ba, 0x27ea41, 0x14412df, 0x67b9b4, 0xdaa51a, 0x1dcb17, 0x4d4afd, 0x6335d5, 0xee2334, 0x17d4e55, 0x1b8b0f0, 0x14999e3, 0x1513dfa, 0x765cf2, 0x56af90, 0x12e16ac, 0x1d3d86c, 0xff279b, 0x18822dd, 0x99d478, 0x8dc0d2, 0x34b666, 0xcf9526, 0x186443d, 0x7a8e29, 0x19c6aa5, 0x1f2a27d, 0x12b2136, 0xd0cd0d, 0x12cb320, 0x17ddb0b, 0x05353b, 0x15b2caf, 0x1e5a507, 0x120f1e5, 0x114605a, 0x14efe4c, 0x568134, 0x11b9f92, 0x174d2a7, 0x692b1d, 0x39e4fe, 0xaaff3d, 0x96224c, 0x13c9f77, 0x110ee8f, 0xf17bea, 0x99fb5d, 0x337141, 0x02b54d, 0x1233a70]
    t25h9=[0x155cbf1, 0x1e4d1b6, 0x17b0b68, 0x1eac9cd, 0x12e14ce, 0x3548bb, 0x7757e6, 0x1065dab, 0x1baa2e7, 0xdea688, 0x81d927, 0x51b241, 0xdbc8ae, 0x1e50e19, 0x15819d2, 0x16d8282, 0x163e035, 0x9d9b81, 0x173eec4, 0xae3a09, 0x5f7c51, 0x1a137fc, 0xdc9562, 0x1802e45, 0x1c3542c, 0x870fa4, 0x914709, 0x16684f0, 0xc8f2a5, 0x833ebb, 0x59717f, 0x13cd050, 0xfa0ad1, 0x1b763b0, 0xb991ce]
    t36h11=[0xd5d628584, 0xd97f18b49, 0xdd280910e, 0xe479e9c98, 0xebcbca822, 0xf31dab3ac, 0x056a5d085,
            0x10652e1d4, 0x22b1dfead, 0x265ad0472, 0x34fe91b86, 0x3ff962cd5, 0x43a25329a, 0x474b4385f, 0x4e9d243e9, 0x5246149ae, 0x5997f5538, 0x683bb6c4c, 0x6be4a7211, 0x7e3158eea, 0x81da494af, 0x858339a74, 0x8cd51a5fe,
            0x9f21cc2d7, 0xa2cabc89c, 0xadc58d9eb, 0xb16e7dfb0, 0xb8c05eb3a, 0xd25ef139d, 0xd607e1962, 0xe4aba3076, 0x2dde6a3da, 0x43d40c678, 0x5620be351, 0x64c47fa65, 0x686d7002a, 0x6c16605ef, 0x6fbf50bb4, 0x8d06d39dc,
            0x9f53856b5, 0xadf746dc9, 0xbc9b084dd, 0xd290aa77b, 0xd9e28b305, 0xe4dd5c454, 0xfad2fe6f2, 0x181a8151a, 0x26be42c2e, 0x2e10237b8, 0x405cd5491, 0x7742eab1c, 0x85e6ac230, 0x8d388cdba, 0x9f853ea93, 0xc41ea2445,
            0xcf1973594, 0x14a34a333, 0x31eacd15b, 0x6c79d2dab, 0x73cbb3935, 0x89c155bd3, 0x8d6a46198, 0x91133675d, 0xa708d89fb, 0xae5ab9585, 0xb9558a6d4, 0xb98743ab2, 0xd6cec68da, 0x1506bcaef, 0x4becd217a, 0x4f95c273f,
            0x658b649dd, 0xa76c4b1b7, 0xecf621f56, 0x1c8a56a57, 0x3628e92ba, 0x53706c0e2, 0x5e6b3d231, 0x7809cfa94, 0xe97eead6f, 0x5af40604a, 0x7492988ad, 0xed5994712, 0x5eceaf9ed, 0x7c1632815, 0xc1a0095b4, 0xe9e25d52b,
            0x3a6705419, 0xa8333012f, 0x4ce5704d0, 0x508e60a95, 0x877476120, 0xa864e950d, 0xea45cfce7, 0x19da047e8, 0x24d4d5937, 0x6e079cc9b, 0x99f2e11d7, 0x33aa50429, 0x499ff26c7, 0x50f1d3251, 0x66e7754ef, 0x96ad633ce,
            0x9a5653993, 0xaca30566c, 0xc298a790a, 0x8be44b65d, 0xdc68f354b, 0x16f7f919b, 0x4dde0e826, 0xd548cbd9f, 0xe0439ceee, 0xfd8b1fd16, 0x76521bb7b, 0xd92375742, 0xcab16d40c, 0x730c9dd72, 0xad9ba39c2, 0xb14493f87,
            0x52b15651f, 0x185409cad, 0x77ae2c68d, 0x94f5af4b5, 0x0a13bad55, 0x61ea437cd, 0xa022399e2, 0x203b163d1, 0x7bba8f40e, 0x95bc9442d, 0x41c0b5358, 0x8e9c6cc81, 0x0eb549670, 0x9da3a0b51, 0xd832a67a1, 0xdcd4350bc,
            0x4aa05fdd2, 0x60c7bb44e, 0x4b358b96c, 0x067299b45, 0xb9c89b5fa, 0x6975acaea, 0x62b8f7afa, 0x33567c3d7, 0xbac139950, 0xa5927c62a, 0x5c916e6a4, 0x260ecb7d5, 0x29b7bbd9a, 0x903205f26, 0xae72270a4, 0x3d2ec51a7,
            0x82ea55324, 0x11a6f3427, 0x1ca1c4576, 0xa40c81aef, 0xbddccd730, 0x0e617561e, 0x969317b0f, 0x67f781364, 0x610912f96, 0xb2549fdfc, 0x06e5aaa6b, 0xb6c475339, 0xc56836a4d, 0x844e351eb, 0x4647f83b4, 0x0908a04f5,
            0x7f51034c9, 0xaee537fca, 0x5e92494ba, 0xd445808f4, 0x28d68b563, 0x04d25374b, 0x2bc065f65, 0x96dc3ea0c, 0x4b2ade817, 0x07c3fd502, 0xe768b5caf, 0x17605cf6c, 0x182741ee4, 0x62846097c, 0x72b5ebf80, 0x263da6e13,
            0xfa841bcb5, 0x7e45e8c69, 0x653c81fa0, 0x7443b5e70, 0x0a5234afd, 0x74756f24e, 0x157ebf02a, 0x82ef46939, 0x80d420264, 0x2aeed3e98, 0xb0a1dd4f8, 0xb5436be13, 0x7b7b4b13b, 0x1ce80d6d3, 0x16c08427d, 0xee54462dd,
            0x1f7644cce, 0x9c7b5cc92, 0xe369138f8, 0x5d5a66e91, 0x485d62f49, 0xe6e819e94, 0xb1f340eb5, 0x09d198ce2, 0xd60717437, 0x0196b856c, 0xf0a6173a5, 0x12c0e1ec6, 0x62b82d5cf, 0xad154c067, 0xce3778832, 0x6b0a7b864,
            0x4c7686694, 0x5058ff3ec, 0xd5e21ea23, 0x9ff4a76ee, 0x9dd981019, 0x1bad4d30a, 0xc601896d1, 0x973439b48, 0x1ce7431a8, 0x57a8021d6, 0xf9dba96e6, 0x83a2e4e7c, 0x8ea585380, 0xaf6c0e744, 0x875b73bab, 0xda34ca901,
            0x2ab9727ef, 0xd39f21b9a, 0x8a10b742f, 0x5f8952dba, 0xf8da71ab0, 0xc25f9df96, 0x06f8a5d94, 0xe42e63e1a, 0xb78409d1b, 0x792229add, 0x5acf8c455, 0x2fc29a9b0, 0xea486237b, 0xb0c9685a0, 0x1ad748a47, 0x03b4712d5,
            0xf29216d30, 0x8dad65e49, 0x0a2cf09dd, 0x0b5f174c6, 0xe54f57743, 0xb9cf54d78, 0x4a312a88a, 0x27babc962, 0xb86897111, 0xf2ff6c116, 0x82274bd8a, 0x97023505e, 0x52d46edd1, 0x585c1f538, 0xbddd00e43, 0x5590b74df,
            0x729404a1f, 0x65320855e, 0xd3d4b6956, 0x7ae374f14, 0x2d7a60e06, 0x315cd9b5e, 0xfd36b4eac, 0xf1df7642b, 0x55db27726, 0x8f15ebc19, 0x992f8c531, 0x62dea2a40, 0x928275cab, 0x69c263cb9, 0xa774cca9e, 0x266b2110e,
            0x1b14acbb8, 0x624b8a71b, 0x1c539406b, 0x3086d529b, 0x0111dd66e, 0x98cd630bf, 0x8b9d1ffdc, 0x72b2f61e7, 0x9ed9d672b, 0x96cdd15f3, 0x6366c2504, 0x6ca9df73a, 0xa066d60f0, 0xe7a4b8add, 0x8264647ef, 0xaa195bf81,
            0x9a3db8244, 0x014d2df6a, 0x0b63265b7, 0x2f010de73, 0x97e774986, 0x248affc29, 0xfb57dcd11, 0x0b1a7e4d9, 0x4bfa2d07d, 0x54e5cdf96, 0x4c15c1c86, 0xcd9c61166, 0x499380b2a, 0x540308d09, 0x8b63fe66f, 0xc81aeb35e,
            0x86fe0bd5c, 0xce2480c2a, 0x1ab29ee60, 0x8048daa15, 0xdbfeb2d39, 0x567c9858c, 0x2b6edc5bc, 0x2078fca82, 0xadacc22aa, 0xb92486f49, 0x51fac5964, 0x691ee6420, 0xf63b3e129, 0x39be7e572, 0xda2ce6c74, 0x20cf17a5c,
            0xee55f9b6e, 0xfb8572726, 0xb2c2de548, 0xcaa9bce92, 0xae9182db3, 0x74b6e5bd1, 0x137b252af, 0x51f686881, 0xd672f6c02, 0x654146ce4, 0xf944bc825, 0xe8327f809, 0x76a73fd59, 0xf79da4cb4, 0x956f8099b, 0x7b5f2655c,
            0xd06b114a6, 0xd0697ca50, 0x27c390797, 0xbc61ed9b2, 0xcc12dd19b, 0xeb7818d2c, 0x092fcecda, 0x89ded4ea1, 0x256a0ba34, 0xb6948e627, 0x1ef6b1054, 0x8639294a2, 0xeda3780a4, 0x39ee2af1d, 0xcd257edc5, 0x2d9d6bc22,
            0x121d3b47d, 0x37e23f8ad, 0x119f31cf6, 0x2c97f4f09, 0xd502abfe0, 0x10bc3ca77, 0x53d7190ef, 0x90c3e62a6, 0x7e9ebf675, 0x979ce23d1, 0x27f0c98e9, 0xeafb4ae59, 0x7ca7fe2bd, 0x1490ca8f6, 0x9123387ba, 0xb3bc73888,
            0x3ea87e325, 0x4888964aa, 0xa0188a6b9, 0xcd383c666, 0x40029a3fd, 0xe1c00ac5c, 0x39e6f2b6e, 0xde664f622, 0xe979a75e8, 0x7c6b4c86c, 0xfd492e071, 0x8fbb35118, 0x40b4a09b7, 0xaf80bd6da, 0x70e0b2521, 0x2f5c54d93,
            0x3f4a118d5, 0x09c1897b9, 0x079776eac, 0x084b00b17, 0x3a95ad90e, 0x28c544095, 0x39d457c05, 0x7a3791a78, 0xbb770e22e, 0x9a822bd6c, 0x68a4b1fed, 0xa5fd27b3b, 0x0c3995b79, 0xd1519dff1, 0x8e7eee359, 0xcd3ca50b1,
            0xb73b8b793, 0x57aca1c43, 0xec2655277, 0x785a2c1b3, 0x75a07985a, 0xa4b01eb69, 0xa18a11347, 0xdb1f28ca3, 0x877ec3e25, 0x31f6341b8, 0x1363a3a4c, 0x075d8b9ba, 0x7ae0792a9, 0xa83a21651, 0x7f08f9fb5, 0x0d0cf73a9,
            0xb04dcc98e, 0xf65c7b0f8, 0x65ddaf69a, 0x2cf9b86b3, 0x14cb51e25, 0xf48027b5b, 0x0ec26ea8b, 0x44bafd45c, 0xb12c7c0c4, 0x959fd9d82, 0xc77c9725a, 0x48a22d462, 0x8398e8072, 0xec89b05ce, 0xbb682d4c9, 0xe5a86d2ff,
            0x358f01134, 0x8556ddcf6, 0x67584b6e2, 0x11609439f, 0x08488816e, 0xaaf1a2c46, 0xf879898cf, 0x8bbe5e2f7, 0x101eee363, 0x690f69377, 0xf5bd93cd9, 0xcea4c2bf6, 0x9550be706, 0x2c5b38a60, 0xe72033547, 0x4458b0629,
            0xee8d9ed41, 0xd2f918d72, 0x78dc39fd3, 0x8212636f6, 0x7450a72a7, 0xc4f0cf4c6, 0x367bcddcd, 0xc1caf8cc6, 0xa7f5b853d, 0x9d536818b, 0x535e021b0, 0xa7eb8729e, 0x422a67b49, 0x929e928a6, 0x48e8aefcc, 0xa9897393c,
            0x5eb81d37e, 0x1e80287b7, 0x34770d903, 0x2eef86728, 0x59266ccb6, 0x0110bba61, 0x1dfd284ef, 0x447439d1b, 0xfece0e599, 0x9309f3703, 0x80764d1dd, 0x353f1e6a0, 0x2c1c12dcc, 0xc1d21b9d7, 0x457ee453e, 0xd66faf540,
            0x44831e652, 0xcfd49a848, 0x9312d4133, 0x3f097d3ee, 0x8c9ebef7a, 0xa99e29e88, 0x0e9fab22c, 0x4e748f4fb, 0xecdee4288, 0xabce5f1d0, 0xc42f6876c, 0x7ed402ea0, 0xe5c4242c3, 0xd5b2c31ae, 0x286863be6, 0x160444d94,
            0x5f0f5808e, 0xae3d44b2a, 0x9f5c5d109, 0x8ad9316d7, 0x3422ba064, 0x2fed11d56, 0xbea6e3e04, 0x04b029eec, 0x6deed7435, 0x3718ce17c, 0x55857f5e2, 0x2edac7b62, 0x085d6c512, 0xd6ca88e0f, 0x2b7e1fc69, 0xa699d5c1b,
            0xf05ad74de, 0x4cf5fb56d, 0x5725e07e1, 0x72f18a2de, 0x1cec52609, 0x48534243c, 0x2523a4d69, 0x35c1b80d1, 0xa4d7338a7, 0x0db1af012, 0xe61a9475d, 0x05df03f91, 0x97ae260bb, 0x32d627fef, 0xb640f73c2, 0x45a1ac9c6,
            0x6a2202de1, 0x57d3e25f2, 0x5aa9f986e, 0x0cc859d8a, 0xe3ec6cca8, 0x54e95e1ae, 0x446887b06, 0x7516732be, 0x3817ac8f5, 0x3e26d938c, 0xaa81bc235, 0xdf387ca1b, 0x0f3a3b3f2, 0xb4bf69677, 0xae21868ed, 0x81e1d2d9d,
            0xa0a9ea14c, 0x8eee297a9, 0x4740c0559, 0xe8b141837, 0xac69e0a3d, 0x9ed83a1e1, 0x5edb55ecb, 0x07340fe81, 0x50dfbc6bf, 0x4f583508a, 0xcb1fb78bc, 0x4025ced2f, 0x39791ebec, 0x53ee388f1, 0x7d6c0bd23, 0x93a995fbe,
            0x8a41728de, 0x2fe70e053, 0xab3db443a, 0x1364edb05, 0x47b6eeed6, 0x12e71af01, 0x52ff83587, 0x3a1575dd8, 0x3feaa3564, 0xeacf78ba7, 0x0872b94f8, 0xda8ddf9a2, 0x9aa920d2b, 0x1f350ed36, 0x18a5e861f, 0x2c35b89c3,
            0x3347ac48a, 0x7f23e022e, 0x2459068fb, 0xe83be4b73]

    TagFamilies = { '16h5':   (t16h5,16),
                    '25h7':   (t25h7,25),
                    '25h9':   (t25h9,25),
                    '36h11':  (t36h11,36)}

    def __init__(self, chosenTagFamiliy):
        try:
            self.chosenTagFamiliy = chosenTagFamiliy
            self.tagCodes = AprilTagCodes.TagFamilies[chosenTagFamiliy][0]
            self.totalBits = AprilTagCodes.TagFamilies[chosenTagFamiliy][1]
        except:
            print("[ERROR]: Unknown tag familiy.")
            sys.exit(0)

#borderBits must be consitent with the variable "blackBorder" in the detector code in file ethz_apriltag2/src/TagFamily.cc
def generateAprilTag(canvas, position, metricSize, tagSpacing, tagID, tagFamililyData, rotation=2, symmCorners=True, borderBits=2):
    #get the tag code
    try:
        tagCode=tagFamililyData.tagCodes[tagID]
    except:
        print("[ERROR]: Requested tag ID of {0} not available in the {1} TagFamiliy".format(tagID, tagFamililyData.chosenTagFamiliy))

    #calculate the bit size of the tag
    sqrtBits = (math.sqrt(tagFamililyData.totalBits))
    bitSquareSize = metricSize / (sqrtBits+borderBits*2)

    #position of tag
    xPos = position[0]
    yPos = position[1]

    #borders (2x bit size)
    borderSize = borderBits*bitSquareSize

    c.fill(path.rect(xPos, yPos, metricSize, borderSize),[color.rgb.black]) #bottom
    c.fill(path.rect(xPos, yPos+metricSize-borderSize, metricSize, borderSize),[color.rgb.black]) #top
    c.fill(path.rect(xPos+metricSize-borderSize, yPos, borderSize, metricSize),[color.rgb.black]) #left
    c.fill(path.rect(xPos, yPos, borderSize, metricSize),[color.rgb.black]) #right

    #create numpy matrix of code
    codeMatrix = np.zeros((int(sqrtBits), int(sqrtBits)))
    for i in range(0, int(sqrtBits)):
        for j in range(0, int(sqrtBits)):
            if not tagCode & (1 << int(sqrtBits)*i+j):
                codeMatrix[i,j] = 1

    #rotation
    codeMatrix = np.rot90(codeMatrix, rotation)

    #bits
    for i in range(0, int(sqrtBits)):
        for j in range(0, int(sqrtBits)):
            if codeMatrix[i,j]:
                c.fill(path.rect(xPos+(j+borderBits)*bitSquareSize, yPos+((borderBits-1)+sqrtBits-i)*bitSquareSize, bitSquareSize, bitSquareSize),[color.rgb.black])

    #add squares to make corners symmetric (decreases the effect of motion blur in the subpix refinement...)
    if symmCorners:
        metricSquareSize = tagSpacing*metricSize

        corners = [
            [xPos-metricSquareSize, yPos-metricSquareSize ],
            [xPos+metricSize, yPos-metricSquareSize],
            [xPos+metricSize, yPos+metricSize],
            [xPos-metricSquareSize, yPos+metricSize]
        ]

        for point in corners:
            c.fill(path.rect(point[0], point[1], metricSquareSize, metricSquareSize),[color.rgb.black])

#tagSpaceing in % of tagSize
def generateAprilBoard(canvas, n_cols, n_rows, name, offset, tagSize, tagSpacing=0.25, tagFamily="t36h11", border=2):

    if(tagSpacing<0 or tagSpacing>1.0):
        print("[ERROR]: Invalid tagSpacing specified.  [0-1.0] of tagSize")
        sys.exit(0)

    #convert to cm
    tagSize = tagSize*100.0

    #get the tag familiy data
    tagFamilyData = AprilTagCodes(tagFamily)

    #create one tag
    numTags = n_cols*n_rows

    #draw tags
    for y in range(0,n_rows):
        for x in range(0,n_cols):
            id = n_cols * y + x + offset
            pos = ( x*(1+tagSpacing)*tagSize, y*(1+tagSpacing)*tagSize)
            generateAprilTag(canvas, pos, tagSize, tagSpacing, id, tagFamilyData, rotation=2, borderBits=border)
            #c.text(pos[0]+0.45*tagSize, pos[1]-0.7*tagSize*tagSpacing, "{0}".format(id))

    #draw axis
    pos = ( -1.5*tagSpacing*tagSize, -1.5*tagSpacing*tagSize)
    c.stroke(path.line(pos[0], pos[1], pos[0]+tagSize*0.3, pos[1]),
             [color.rgb.red,
              deco.earrow([deco.stroked([color.rgb.red, style.linejoin.round]),
                           deco.filled([color.rgb.red])], size=tagSize*0.10)])
    c.text(pos[0]+tagSize*0.3, pos[1], "x")

    c.stroke(path.line(pos[0], pos[1], pos[0], pos[1]+tagSize*0.3),
             [color.rgb.green,
              deco.earrow([deco.stroked([color.rgb.green, style.linejoin.round]),
                           deco.filled([color.rgb.green])], size=tagSize*0.10)])
    c.text(pos[0], pos[1]+tagSize*0.3, "y")

    #text
    caption = "%dx%d tags, size=%.2fcm, spacing=%.2fcm, id offset=%d, %s" % (n_cols,n_rows,tagSize,tagSpacing*tagSize, offset, name)
    c.text(pos[0]+0.6*tagSize, pos[0], caption)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a PDF with a calibration pattern.')
    parser.add_argument('--double', action='store_true', help='Create a double sided marker.')

    parser.add_argument('--nx', type=int, default=6, dest='n_cols', help='The number of tags in x direction (default: %(default)s)\n')
    parser.add_argument('--ny', type=int, default=7, dest='n_rows', help='The number of tags in y direction (default: %(default)s)')

    parser.add_argument('--tsize', type=float, default=0.02, dest='tsize', help='The size of one tag [m] (default: %(default)s)')
    parser.add_argument('--tspace', type=float, default=0.25, dest='tagspacing', help='The space between the tags in fraction of the edge size [0..1] (default: %(default)s)')
    parser.add_argument('--tfam', default='36h11', dest='tagfamily', help='Family of April tags {0} (default: %(default)s)'.format(AprilTagCodes.TagFamilies.keys()))
    parser.add_argument('--border', type=int, default=2, help='Size of border for each tag.')

    parsed = parser.parse_args()

    output_name_pdf = 'tags/marker_%%s_%sb%d_%dx%d_%dcm' % (parsed.tagfamily, parsed.border, parsed.n_rows, parsed.n_cols, parsed.tsize * 100)
    output_name_json = 'tags/marker_%sb%d_%dx%dx_%dcm.json' % (parsed.tagfamily, parsed.border, parsed.n_rows, parsed.n_cols, parsed.tsize*100)
    assert not os.path.exists(output_name_json), 'Output marker does already exist!'

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # parse the argument list
    try:
        parsed = parser.parse_args()

    except:
        sys.exit(0)

    if parsed.double:
        jobs = [(0, 'front'),
                (parsed.n_cols*parsed.n_rows, 'back')]

    else:
        jobs = [(0, 'front')]

    for offset, name in jobs:
        # open a new canvas
        c = canvas.canvas()

        # draw the board
        generateAprilBoard(canvas, parsed.n_cols, parsed.n_rows, name, offset,
                           parsed.tsize, parsed.tagspacing, parsed.tagfamily, parsed.border)

        # write to file
        my_mkdir(output_name_pdf % name, is_file=True)
        c.writePDFfile(output_name_pdf % name)
        print('Created %s.pdf' % output_name_pdf % name)

    tag_desc = dict()
    tag_desc['family'] = parsed.tagfamily
    tag_desc['border'] = parsed.border
    tag_desc['double'] = parsed.double
    tag_desc['tsize'] = parsed.tsize
    tag_desc['tspace'] = parsed.tsize*parsed.tagspacing  # actual size in meters
    tag_desc['n_x'] = parsed.n_cols
    tag_desc['n_y'] = parsed.n_rows
    tag_desc['offset'] = [0.0, 0.0, 0.0]  # these we dont know because they depend on the manufacturing process

    json_dump(output_name_json, tag_desc)
    print('Created %s' % output_name_json)





