import os
import click
from tqdm import tqdm


@click.command()
@click.option("--datadir", prompt="data folder contains .dat s", help="data folder contains .dat s")
def convert(datadir):
	"""
	convert the full dataset mitdb (data and annotatiosn) to csv and ann files.

	:param datadir: Str, eg '/home/congyu/dataset/ECG/mitbih/'
	:return:
	"""
	current_dir = os.getcwd()
	os.chdir(datadir)

	# Create folder
	datadir_out = datadir + 'csv/'
	if not os.path.exists(datadir_out):
		os.mkdir(datadir_out)

	ptids = [f[:-4] for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f)) if (f.find('.dat') != -1)]
	# patient_ids = ["101", "102", ...]
	# print(records)

	for ptid in tqdm(ptids):
		command = 'rdsamp -r {ptid} -c -H -f 0 -v > {datadir_out}{ptid}.csv'.format(ptid=ptid, datadir_out=datadir_out)
		os.system(command)

		command = 'rdann -r {ptid} -f 0 -a atr -v  > {datadir_out}{ptid}.ann'.format(ptid=ptid, datadir_out=datadir_out)
		os.system(command)

	os.chdir(current_dir)


if __name__ == '__main__':
	convert()
