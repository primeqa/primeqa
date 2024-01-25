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
parser.add_argument('-P', '--apikey', type=str, default=None,
                    help='The password to use.')
parser.add_argument('-S', '--server', default=None, type=str,
                    help='The server to connect to.')
# Parse the arguments
args = parser.parse_args()

server = None
if args.server == 'ailang':
    args.apikey = os.environ['AILANG_API_KEY']
    args.fingerprint = os.environ['AILANG_SSL_FINGERPRINT']
    if args.host is None:
        args.host = os.environ['ailang']
elif args.server == 'convai':
    args.apikey = os.environ['ES_API_KEY']
    args.fingerprint = os.environ['ES_SSL_FINGERPRINT']
    if args.host is None:
        args.host = os.environ['convai']
elif args.server == 'resconvai':
    args.apikey = os.environ['RESCONVAI_API_KEY']
    args.fingerprint = os.environ['RESCONVAI_SSL_FINGERPRINT']
    if args.host is None:
        args.host = os.environ['resconvai']
elif args.server == 'localhost':
    args.apikey = os.environ['LOCAL_API_KEY']
    args.fingerprint = os.environ['LOCAL_SSL_FINGERPRINT']
    if args.host is None:
        args.host = "localhost"
else:
    print('Unknown server')
    sys.exit(10)

es_server = f"https://{args.host}:9200"

client = Elasticsearch(es_server,
                       ssl_assert_fingerprint=(args.fingerprint),
                       api_key=args.apikey
                       )

print(client.info())