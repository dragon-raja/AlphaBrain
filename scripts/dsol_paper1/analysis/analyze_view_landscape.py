"""Plot/analyze an existing full-view panel; never launches model inference."""
import argparse
from AlphaBrain.research.dsol.analysis.view_landscape import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--population', help='Optional original 400-initial P0 analysis directory')
    args = parser.parse_args()
    result = run(args.root, args.output, population=args.population)
    print(result['scope'])
