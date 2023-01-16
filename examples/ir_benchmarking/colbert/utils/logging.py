import os
import sys
import ujson
import traceback

from colbert.utils.utils import print_message, create_directory


class Logger():
    def __init__(self, rank, run):
        self.rank = rank
        self.is_main = self.rank in [-1, 0]
        self.run = run
        self.logs_path = os.path.join(self.run.path, "logs/")

        if self.is_main:
            create_directory(self.logs_path)

    def _log_exception(self, etype, value, tb):
        if not self.is_main:
            return

        output_path = os.path.join(self.logs_path, 'exception.txt')
        trace = ''.join(traceback.format_exception(etype, value, tb)) + '\n'
        print_message(trace, '\n\n')

        self.log_new_artifact(output_path, trace)

    def _log_all_artifacts(self):
        if not self.is_main:
            return

    def _log_args(self, args):
        if not self.is_main:
            return
        
        with open(os.path.join(self.logs_path, 'args.txt'), 'w') as output_metadata:
            output_metadata.write(' '.join(sys.argv) + '\n')

    def log_metric(self, name, value, step, log_to_mlflow=True):
        if not self.is_main:
            return


    def log_new_artifact(self, path, content):
        with open(path, 'w') as f:
            f.write(content)


    def warn(self, *args):
        msg = print_message('[WARNING]', '\t', *args)

        with open(os.path.join(self.logs_path, 'warnings.txt'), 'a') as output_metadata:
            output_metadata.write(msg + '\n\n\n')

    def info_all(self, *args):
        print_message('[' + str(self.rank) + ']', '\t', *args)

    def info(self, *args):
        if self.is_main:
            print_message(*args)
