import torch


def calc_input_channels(input_list):
    channels = 0
    for item in input_list:
        if item == "pose":
            channels += 3
        elif item == "appearance":
            channels += 3
        elif item == "target":
            channels += 3
        else:
            assert False, (
                "items in input list have to be in ['pose', 'appearance', 'target'], not "
                + item
            )
    return channels


def cat_inputs(input_dict, input_list):
    return torch.cat([input_dict[key] for key in input_list], dim=1)
