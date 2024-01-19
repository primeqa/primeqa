import sys

from elasticsearch import Elasticsearch;
import os

from argparse import ArgumentParser
parser = ArgumentParser(description='Process host and password.')

# Add the arguments
parser.add_argument('-H', '--host', default=None, type=str,
                    help='The host to connect to.')
parser.add_argument('-F', '--fingerprint', default=None, type=str,
                    help='The fingerprint')
parser.add_argument('-P', '--password', type=str, default=None,
                    help='The password to use.')
parser.add_argument('-S', '--server', default=None, type=str,
                    help='The server to connect to.')
# Parse the arguments
args = parser.parse_args()

server = None
if args.server == 'ailang':
    args.password = os.environ['AILANG_PASSWORD']
    args.fingerprint = os.environ['AILANG_SSL_FINGERPRINT']
    if args.host is None:
        args.host = os.environ['ailang']
elif args.server == 'convai':
    args.password = os.environ['ES_PASSWORD']
    args.fingerprint = os.environ['ES_SSL_FINGERPRINT']
    if args.host is None:
        args.host = os.environ['convai']
elif args.server == 'resconvai':
    args.password = os.environ['RESCONVAI_PASSWORD']
    args.fingerprint = os.environ['RESCONVAI_SSL_FINGERPRINT']
    if args.host is None:
        args.host = os.environ['resconvai']
else:
    print('Unknown server')
    sys.exit(10)


client = Elasticsearch(f"https://{args.host}:9200",
                       ssl_assert_fingerprint=(args.fingerprint),
                       api_key=args.password
                       )

client.info()