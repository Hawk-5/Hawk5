# importing the required module 
import matplotlib.pyplot as plt 

# The
# x = [739, 740, 732, 709, 688, 671, 652, 634, 618, 606, 598, 595, 601, 626, 663, 700, 729, 755, 766, 793, 829, 859, 893, 921, 935, 941, 936, 932, 969]
# y = [153, 152, 157, 168, 180, 188, 207, 226, 246, 255, 259, 260, 258, 249, 245, 242, 229, 213, 206, 201, 196, 194, 193, 189, 182, 167, 163, 155, 180]

# x1 = [726, 727, 729, 727, 720, 702, 681, 657, 637, 630, 613, 605, 607, 632, 675, 713, 744, 777, 813, 840, 870, 888, 901, 919, 935, 944, 949]
# y1 = [153, 158, 160, 163, 171, 183, 195, 214, 223, 229, 247, 257, 256, 247, 228, 213, 207, 199, 189, 185, 170, 166, 167, 163, 158, 164, 169]

# x2 = [759, 761, 748, 752, 762, 757, 747, 713, 682, 654, 627, 611, 612, 615, 619, 641, 682, 724, 772, 811, 848, 867, 896, 914, 922, 921, 932, 904]
# y2 = [150, 153, 160, 174, 181, 181, 183, 197, 208, 223, 243, 257, 256, 249, 245, 235, 215, 208, 207, 201, 195, 191, 185, 173, 172, 174, 173, 149]

# # Hospital 
# x = [588, 558, 575, 562, 567, 508, 445, 402, 367, 350, 348, 358, 399, 477, 548, 591, 649, 679, 726, 764, 802, 835, 878, 929, 964, 984, 997, 997, 986, 944, 869, 813, 730, 700, 644, 592, 551, 510, 465, 422, 370, 341, 284, 232, 215, 209, 234, 279, 352, 412, 430, 446, 436, 426, 424, 439, 476, 509, 545, 588, 615, 639, 662, 682, 702, 718, 725, 739, 757, 781, 804, 828, 859, 895, 937, 979, 1012, 1026, 1033, 1044, 1054, 1067, 1082, 1087, 1075, 1034, 988, 965, 923, 880, 835, 790, 764, 715, 683, 647, 623, 591, 561, 529, 508, 481, 450, 415, 385, 363, 342, 337, 322, 317, 316, 305]
# y = [249, 289, 292, 285, 280, 247, 213, 202, 197, 193, 194, 191, 194, 208, 217, 225, 237, 244, 254, 258, 261, 264, 271, 268, 269, 269, 268, 265, 262, 257, 246, 234, 219, 214, 201, 193, 185, 176, 173, 164, 157, 156, 150, 142, 144, 148, 150, 150, 150, 142, 136, 143, 144, 147, 145, 148, 155, 157, 159, 162, 159, 158, 161, 161, 163, 169, 169, 176, 185, 200, 213, 220, 220, 222, 227, 229, 236, 240, 240, 239, 244, 247, 248, 252, 247, 246, 244, 242, 238, 238, 238, 237, 236, 234, 234, 240, 244, 243, 238, 233, 230, 231, 235, 241, 243, 243, 245, 247, 250, 252, 251, 247]

# x1 = [626, 622, 614, 615, 594, 570, 497, 466, 446, 422, 409, 402, 395, 385, 375, 354, 341, 346, 370, 404, 433, 471, 516, 567, 613, 655, 687, 729, 770, 821, 851, 875, 903, 917, 926, 945, 969, 983, 992, 1004, 992, 965, 947, 933, 906, 869, 833, 807, 779, 760, 726, 703, 662, 607, 552, 515, 477, 444, 410, 391, 379, 344, 329, 315, 302, 282, 269, 272, 277, 318, 348, 365, 388, 395, 402, 404, 406, 422, 435, 440, 457, 490, 522, 544, 562, 582, 606, 633, 646, 662, 686, 699, 719, 729, 730, 720, 732, 771, 805, 818, 846, 870, 898, 928, 957, 982, 995, 1008, 1025, 1043, 1050, 1061, 1073, 1085, 1093, 1099, 1089, 1070, 1046, 1005, 976, 955, 931, 874, 832, 785, 741, 703, 674, 636, 617, 591, 578, 547, 523, 488, 454, 405, 370, 357, 340, 326, 324]
# y1 = [256, 265, 257, 247, 250, 246, 226, 219, 215, 208, 203, 187, 180, 170, 162, 150, 148, 149, 154, 167, 176, 185, 191, 190, 194, 195, 200, 204, 211, 223, 231, 235, 241, 245, 246, 248, 251, 248, 248, 252, 249, 243, 240, 240, 243, 236, 224, 216, 208, 200, 197, 207, 206, 201, 191, 189, 185, 183, 177, 173, 169, 168, 168, 168, 170, 170, 168, 165, 169, 169, 167, 169, 172, 171, 172, 171, 169, 162, 159, 158, 159, 164, 165, 163, 164, 158, 163, 167, 164, 165, 165, 165, 166, 164, 162, 162, 165, 178, 194, 198, 200, 201, 205, 207, 214, 227, 228, 224, 229, 226, 228, 234, 235, 239, 246, 255, 253, 252, 254, 248, 244, 247, 254, 248, 245, 246, 239, 239, 240, 237, 237, 235, 235, 237, 233, 242, 246, 252, 262, 263, 264, 262, 250]

# x2 = [566, 563, 566, 568, 567, 573, 584, 587, 584, 578, 572, 555, 533, 499, 477, 444, 431, 413, 396, 381, 364, 352, 342, 367, 380, 408, 450, 493, 550, 606, 675, 717, 759, 787, 814, 852, 885, 915, 941, 954, 964, 974, 983, 979, 975, 969, 948, 913, 892, 860, 807, 778, 734, 700, 655, 634, 601, 560, 528, 496, 456, 431, 416, 390, 362, 341, 321, 302, 292, 280, 268, 256, 267, 303, 356, 381, 399, 413, 412, 423, 433, 439, 456, 471, 482, 501, 526, 544, 553, 574, 599, 639, 664, 689, 708, 710, 711, 718, 748, 779, 796, 818, 847, 877, 885, 907, 924, 940, 953, 969, 993, 1016, 1040, 1051, 1070, 1075, 1076, 1079, 1062, 1034, 971, 916, 852, 824, 796, 765, 728, 698, 667, 643, 621, 584, 542, 497, 465, 437, 412, 400, 382, 377, 370, 341, 334, 327, 310, 269]
# y2 = [284, 283, 283, 284, 287, 288, 284, 274, 269, 267, 264, 256, 236, 213, 206, 197, 195, 191, 185, 177, 171, 170, 166, 175, 180, 181, 183, 186, 193, 205, 214, 219, 222, 221, 224, 229, 230, 240, 246, 249, 253, 256, 258, 259, 255, 254, 250, 246, 248, 248, 245, 234, 225, 219, 209, 206, 205, 206, 204, 199, 196, 191, 187, 187, 187, 187, 186, 183, 181, 168, 168, 168, 163, 164, 165, 160, 157, 155, 151, 157, 157, 157, 162, 165, 167, 169, 173, 173, 171, 167, 166, 159, 151, 148, 147, 145, 147, 153, 163, 172, 178, 185, 197, 210, 214, 225, 226, 231, 231, 232, 242, 246, 252, 255, 266, 261, 259, 263, 258, 253, 253, 254, 255, 251, 247, 248, 244, 237, 234, 236, 236, 237, 245, 243, 241, 245, 242, 242, 247, 241, 245, 250, 238, 240, 245, 227]

# x3 = [611, 604, 596, 602, 608, 613, 603, 588, 537, 513, 462, 414, 383, 367, 338, 332, 341, 353, 359, 371, 447, 530, 580, 636, 687, 743, 790, 834, 863, 887, 908, 929, 947, 958, 972, 974, 967, 944, 893, 815, 765, 700, 661, 605, 562, 527, 492, 457, 420, 393, 347, 312, 290, 283, 280, 299, 333, 381, 421, 436, 447, 446, 445, 455, 519, 571, 599, 612, 641, 659, 669, 700, 727, 754, 774, 825, 873, 896, 926, 961, 989, 1015, 1039, 1061, 1081, 1098, 1076, 1010, 933, 840, 768, 712, 645, 585, 509, 472, 429, 398, 365, 336, 336, 335]
# y3 = [245, 271, 269, 261, 257, 261, 250, 245, 246, 247, 239, 229, 219, 198, 174, 166, 167, 166, 165, 166, 181, 188, 187, 193, 195, 199, 209, 215, 218, 226, 231, 238, 246, 248, 250, 249, 253, 251, 247, 242, 234, 225, 222, 219, 216, 212, 211, 205, 198, 199, 190, 184, 178, 175, 174, 170, 168, 167, 162, 159, 157, 157, 158, 158, 156, 155, 153, 149, 144, 149, 150, 152, 151, 153, 166, 183, 198, 206, 220, 228, 235, 244, 252, 253, 258, 262, 249, 237, 234, 225, 234, 245, 243, 239, 243, 244, 248, 263, 273, 274, 269, 259]

# Mom
x = [430, 430, 427, 422, 411, 403, 386, 370, 387, 384, 381, 377, 395, 335, 386, 380, 378, 372, 384, 405, 425, 432, 443, 453, 452]
y = [353, 352, 343, 326, 303, 286, 262, 242, 228, 215, 201, 184, 179, 163, 157, 155, 167, 183, 211, 244, 270, 290, 311, 352, 358]

x1 = [421, 423, 424, 420, 416, 412, 405, 391, 372, 358, 347, 341, 336, 335, 334, 333, 326, 324, 335, 346, 365, 388, 401, 410, 422, 425, 426, 421, 423, 431, 441, 447, 449, 451, 455, 466]
y1 = [388, 368, 380, 395, 386, 359, 343, 314, 284, 260, 237, 223, 207, 195, 182, 164, 152, 148, 156, 167, 187, 222, 244, 269, 296, 311, 314, 311, 318, 336, 346, 353, 354, 357, 355, 346]

x2 = [454, 455, 453, 449, 445, 431, 414, 406, 397, 385, 376, 372, 363, 360, 360, 354, 353, 349, 351, 357, 360, 376, 385, 391, 405, 415, 420, 423, 425, 426, 429, 431, 439, 447, 447, 450, 455]
y2 = [374, 377, 373, 372, 372, 363, 336, 302, 269, 232, 196, 187, 177, 176, 185, 187, 179, 164, 156, 174, 178, 193, 206, 217, 244, 266, 281, 292, 304, 309, 320, 329, 329, 340, 346, 347, 351]

plt.plot(x, y, label = "line 1")
plt.plot(x1,y1, label = "line 2")
plt.plot(x2,y2,label = "line 3")
# plt.plot(x3,y3,label = "line 4")

plt.xlabel('x - axis')  
plt.ylabel('y - axis') 
 
plt.title('Shape Graph') 

# function to show the plot 
plt.show()