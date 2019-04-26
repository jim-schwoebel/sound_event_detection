import sys, sed_vis, dcase_util 

def visualize_sample(audiofilepath, csvfilepath):
	# taken from sed_vis documentation - https://github.com/TUT-ARG/sed_vis
	# thanks Audio Research Group, Tampere University! 
	
	# Load audio signal first
	audio_container = dcase_util.containers.AudioContainer().load(audiofilepath)

	# Load event lists
	reference_event_list = dcase_util.containers.MetaDataContainer().load(csvfilepath)
	event_lists = {'reference': reference_event_list}

	# Visualize the data
	vis = sed_vis.visualization.EventListVisualizer(event_lists=event_lists,
	                                                audio_signal=audio_container.data,
	                                                sampling_rate=audio_container.fs)
	vis.show()

audiofilepath=sys.argv[1]
csvfilepath=sys.argv[2]

visualize_sample(audiofilepath, csvfilepath)