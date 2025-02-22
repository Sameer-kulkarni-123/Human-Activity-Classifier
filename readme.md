the original data set has been renamed to Action-Recognition-Dataset

the folders inside the Action-Recognition-Dataset/source-images3 have to be renamed to jump_1, jump_2, kick_1, ...etc


DO THE ABOVE STEP AND FOLLOW THE BELOW EXCEPTIONS:KJ


there are 4 folders for "stand" but in valid_images.txt there are 5 entries for stand, stand3 has been repeated twice:

instead of this:

stand_1
55 520

stand_2
60 500

stand_3
35 395

stand_4
481 619

stand_5
40 335

do this:

stand_1
55 520

stand_2
60 500

stand_3
35 395
481 619

stand_4
40 335


AND

walk_1 and walk_2 has to be interchanged