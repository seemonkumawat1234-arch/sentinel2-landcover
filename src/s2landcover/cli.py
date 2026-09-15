"""Command line interface.

    python -m s2landcover demo
    python -m s2landcover supervised --b2 ... --labels labels.tif
    python -m s2landcover cluster --b2 ... --clusters 6
"""

from __future__ import annotations

import argparse
import os
import sys

from .features import BANDS
from .pipeline import plot_result, run_supervised, run_unsupervised
from .scene import LandcoverScene, load_bands, load_labels, synthetic_scene


def _add_band_args(p):
    for band in BANDS:
        p.add_argument("--{}".format(band.lower()), required=True, metavar="TIF",
                       help="{} GeoTIFF".format(band))
    p.add_argument("--scale", type=float, default=10000.0,
                   help="reflectance scale factor; Sentinel-2 L2A is %(default)s")


def _add_common(p):
    p.add_argument("--no-indices", action="store_true",
                   help="use raw bands only, skipping the spectral indices")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--no-figure", action="store_true")


def _bands_from_args(args):
    return {b: getattr(args, b.lower()) for b in BANDS}


def _finish(result, scene, args, stem, title):
    print(result.text_summary())
    if not args.no_figure:
        print("\nfigure: {}".format(plot_result(
            result, scene, os.path.join(args.out_dir, stem + ".png"),
            title=title)))
    for name, path in sorted(result.written.items()):
        print("{:12s} {}".format(name, path))
    return 0


def cmd_demo(args) -> int:
    scene = synthetic_scene(shape=(args.size, args.size), seed=args.seed,
                            noise=args.noise)
    if args.cluster:
        result = run_unsupervised(scene, n_clusters=args.clusters,
                                  use_indices=not args.no_indices,
                                  seed=args.seed, out_dir=args.out_dir)
        return _finish(result, scene, args, "landcover_demo",
                       "k-means clusters, synthetic scene")
    result = run_supervised(scene, use_indices=not args.no_indices,
                            block=args.block, test_fraction=args.test_fraction,
                            n_estimators=args.trees, seed=args.seed,
                            out_dir=args.out_dir)
    return _finish(result, scene, args, "landcover_demo",
                   "Land cover classification, synthetic scene")


def cmd_supervised(args) -> int:
    scene = load_bands(_bands_from_args(args), scale=args.scale)
    labels = load_labels(args.labels, scene.shape)
    scene = LandcoverScene(bands=scene.bands, labels=labels, crs=scene.crs,
                           transform=scene.transform,
                           pixel_size_m=scene.pixel_size_m)
    result = run_supervised(scene, use_indices=not args.no_indices,
                            block=args.block, test_fraction=args.test_fraction,
                            n_estimators=args.trees, seed=args.seed,
                            out_dir=args.out_dir)
    return _finish(result, scene, args, "landcover", "Land cover classification")


def cmd_cluster(args) -> int:
    scene = load_bands(_bands_from_args(args), scale=args.scale)
    result = run_unsupervised(scene, n_clusters=args.clusters,
                              use_indices=not args.no_indices,
                              seed=args.seed, out_dir=args.out_dir)
    return _finish(result, scene, args, "landcover_clusters", "k-means clusters")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="s2landcover",
        description="Land cover classification from Sentinel-2, with a "
                    "spatial-block train/test split and accuracy assessment.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="run on a synthetic scene, no data needed")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--noise", type=float, default=0.032,
                   help="per-band reflectance noise; higher makes the classes "
                        "overlap more (default: %(default)s)")
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--test-fraction", type=float, default=0.35)
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--cluster", action="store_true",
                   help="run k-means instead of the supervised forest")
    p.add_argument("--clusters", type=int, default=6)
    _add_common(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("supervised", help="random forest on real bands + labels")
    _add_band_args(p)
    p.add_argument("--labels", required=True, metavar="TIF",
                   help="rasterised training classes, 0 = unlabelled")
    p.add_argument("--block", type=int, default=16,
                   help="spatial block size in pixels for the train/test split "
                        "(default: %(default)s)")
    p.add_argument("--test-fraction", type=float, default=0.35)
    p.add_argument("--trees", type=int, default=300)
    _add_common(p)
    p.set_defaults(func=cmd_supervised)

    p = sub.add_parser("cluster", help="k-means on real bands, no labels needed")
    _add_band_args(p)
    p.add_argument("--clusters", type=int, default=6)
    _add_common(p)
    p.set_defaults(func=cmd_cluster)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
