import json
import argparse
from tqdm import tqdm

def generate_new_annotations(original_anno):
    """
    remove "no_interaction" from the verb

    """
    # verbs = "verbs": ["adjust", "assemble", "block", "blow", "board", "break", "brush_with", "buy", "carry", "catch", "chase", "check", "clean", "control", "cook", "cut", "cut_with", "direct", "drag", "dribble", "drink_with", "drive", "dry", "eat", "eat_at", "exit", "feed", "fill", "flip", "flush", "fly", "greet", "grind", "groom", "herd", "hit", "hold", "hop_on", "hose", "hug", "hunt", "inspect", "install", "jump", "kick", "kiss", "lasso", "launch", "lick", "lie_on", "lift", "light", "load", "lose", "make", "milk", "move", "no_interaction", "open", "operate", "pack", "paint", "park", "pay", "peel", "pet", "pick", "pick_up", "point", "pour", "pull", "push", "race", "read", "release", "repair", "ride", "row", "run", "sail", "scratch", "serve", "set", "shear", "sign", "sip", "sit_at", "sit_on", "slide", "smell", "spin", "squeeze", "stab", "stand_on", "stand_under", "stick", "stir", "stop_at", "straddle", "swing", "tag", "talk_on", "teach", "text_on", "throw", "tie", "toast", "train", "turn", "type_on", "walk", "wash", "watch", "wave", "wear", "wield", "zip"]
    new_anno = original_anno.copy()
    new_anns = new_anno['annotation']

    emp = original_anno['empty']
    count = 0
    for ann_idx, (anno,img_size) in tqdm(enumerate(zip(original_anno['annotation'], original_anno['size'])), total=len(original_anno['annotation'])):
        bh, bo, hoi, obj, verb = [], [], [], [], []
        for box_h, box_o, hoi_idx, obj_idx, verb_idx in zip(
            anno['boxes_h'], anno['boxes_o'], anno['hoi'], anno['object'], anno['verb']):
            box_w_ = (box_o[2] - box_o[0]) / img_size[0]
            box_h_ = (box_o[3] - box_o[1]) / img_size[1]
            box_area = box_w_ * box_h_
            if box_area < 0.05:  # filter out small boxes
                count += 1
                continue
            if verb_idx == 57: # "no_interaction"
                count += 1
                continue
            bh.append(box_h)
            bo.append(box_o)
            hoi.append(hoi_idx)
            obj.append(obj_idx)
            verb.append(verb_idx)
        assert len(bh) == len(bo) == len(hoi) == len(obj) == len(verb)
        if len(hoi) == 0:
            emp.append(ann_idx)
        new_anns[ann_idx] = {
            'boxes_h': bh,
            'boxes_o': bo,
            'hoi': hoi,
            'object': obj,
            'verb': verb
        }
    new_anno['empty'] = sorted(list(set(emp)))
    new_anno['annotation'] = new_anns
    print(f"Removed {count} annotations with 'no_interaction' or small boxes.")
    return new_anno

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Generate zero-shot annotations')
    argparser.add_argument('--input', type=str, default='', help='Input file path')
    argparser.add_argument('--output', type=str, default='', help='Output file path')
    args = argparser.parse_args()

    with open(args.input, 'r') as f:
        original_anno = json.load(f)
    
    new_anno = generate_new_annotations(original_anno)
    
    with open(args.output, 'w') as f:
        json.dump(new_anno, f)
    breakpoint()