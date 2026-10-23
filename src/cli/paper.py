"""Compatibility entry point. Unsupported legacy paper generator preserved in archive."""
import subprocess
import sys

def setup_parser(subparsers):
    p=subparsers.add_parser('paper')
    p.add_argument('paper_cmd',choices=['build','package'])

def handle_command(args):
    script='build_manuscript.py' if args.paper_cmd=='build' else 'package_revision.py'
    subprocess.run([sys.executable,'scripts/'+script],check=True)
