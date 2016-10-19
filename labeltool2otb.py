import sys, getopt
import numpy as np

def create_otb_anno(file):
	groundTruth = np.loadtxt(file, delimiter=" ")
	groundTruth = groundTruth[groundTruth[:,6]==0,:]
	positions =  groundTruth[:,1:5]
	
	np.savetxt('example.txt', positions, delimiter=',', fmt='%d')

def main(argv):
	try:
		opts, args = getopt.getopt(argv,"hf:",["file="])
	except getopt.GetoptError:
		print 'Error: command example: python labeltool2otb.py -f <pathtofile>'
		sys.exit(2)

	for opt, arg in opts:
		if opt == '-h':
			print 'Command example: python labeltool2otb.py -f <pathtofile>'
			sys.exit()
		elif opt in ("-f", "--file"):
			create_otb_anno(arg)

if __name__ == "__main__":
	main(sys.argv[1:])