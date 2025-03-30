def get_coco_pretrain_from_obj365(cur_tensor, pretrain_tensor):
    """Get coco weights from obj365 pretrained model."""
    if pretrain_tensor.size() == cur_tensor.size():
        return pretrain_tensor
    cur_tensor.requires_grad = False
    coco_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74,
        75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    ]
    obj365_ids = [
        0, 46, 5, 58, 114, 55, 116, 65, 21, 40, 176, 127, 249, 24, 56, 139, 92, 78, 99, 96,
        144, 295, 178, 180, 38, 39, 13, 43, 120, 219, 148, 173, 165, 154, 137, 113, 145, 146,
        204, 8, 35, 10, 88, 84, 93, 26, 112, 82, 265, 104, 141, 152, 234, 143, 150, 97, 2,
        50, 25, 75, 98, 153, 37, 73, 115, 132, 106, 61, 163, 134, 277, 81, 133, 18, 94, 30,
        169, 70, 328, 226
    ]

    for coco_id, obj_id in zip(coco_ids, obj365_ids):
        cur_tensor[coco_id] = pretrain_tensor[obj_id + 1]
    return cur_tensor