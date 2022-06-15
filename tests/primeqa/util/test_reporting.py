from primeqa.util.reporting import Reporting


class Tester:
    def test_reporting(self):
        report = Reporting(gather_samples=('x'))
        report.moving_averages(x=1, y=2)
        report.is_time()
        report.get_count('x')
        report.display()
        report.display_warn()
        report.elapsed_seconds()
        report.elapsed_time_str()
        report.progress_str()
        report.get_moving_average('x')
        report.moving_averages(x=1, y=2)
        report.get_samples('x')
        report.reset()
