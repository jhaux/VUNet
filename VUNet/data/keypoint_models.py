class OPENPOSE_18:
    '''The stickman configuration for OpenPose's keypoint output with 18
    joints.
    See the corresponding
    [page](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#pose-output-format-coco).

    Numbers shown below correspond to numbers in the image on the page linked
    above.
    '''
    LEFT_LINES = [
        (2, 3),
        (3, 4),
        (2, 8),
        (8, 9),
        (9, 10)
    ]
    
    LEFT_POINTS = [2, 3, 4, 8, 9, 10]
    
    RIGHT_LINES = [
        (5, 6),
        (6, 7),
        (5, 11),
        (11, 12),
        (12, 13)
    ]
    
    RIGHT_POINTS = [5, 6, 7, 11, 12, 13]
    
    CENTER_LINES = [
        (16, 14),
        (14, 0),
        (0, 15),
        (15, 17),
        (0, 1),
    ]
    
    CENTER_BODY = [1, 2, 8, 11, 5]
    
    CENTER_POINTS = [0, 1, 14, 15, 16, 17]
