#!/usr/bin/env python3
import os
import argparse

def main():
    ap = argparse.ArgumentParser(description="Generate movie_list from frame subfolders")
    ap.add_argument("--frames-root", default="/media/zhong/mypassport/archive(1)/videos_cut/frames_1s",
                    help="Root dir containing per-movie subfolders")
    ap.add_argument("--out", "-o", default="/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/movies.txt",
                    help="Output movie_list path")
    ap.add_argument("--with-ext", action="store_true",
                    help="Append .mp4 to each name")
    args = ap.parse_args()

    names = []
    for name in os.listdir(args.frames_root):
        p = os.path.join(args.frames_root, name)
        if os.path.isdir(p):
            names.append(name)

    # 数字优先排序
    def keyfn(x):
        return (0, int(x)) if x.isdigit() else (1, x)

    names.sort(key=keyfn)

    with open(args.out, "w") as f:
        for n in names:
            f.write(n + (".mp4" if args.with_ext else "") + "\n")

    print(f"Saved {len(names)} names to {args.out}")

if __name__ == "__main__":
    main()

#python utils/create_movie_liset.py --frames-root "/media/zhong/mypassport/archive(1)/videos_cut/frames_1s" --out /media/zhong/1.0T/zhong_work/zhong_detr/inference_results/movies.txt