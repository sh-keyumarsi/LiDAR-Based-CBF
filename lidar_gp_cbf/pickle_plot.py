
# from scenarios_unicycle.Resilient_pickleplot import preamble_setting, scenario_pkl_plot, exp_video_pkl_plot 
# from scenarios_SI.AgriControl2022_pickleplot import preamble_setting, scenario_pkl_plot, exp_video_pkl_plot
from scenarios.obstacle_GP_pickleplot import preamble_setting, scenario_pkl_plot, exp_video_pkl_plot 

# Basic script for post-processing

def main():
    preamble_setting()
    scenario_pkl_plot()

    # Uncomment this part to make a snapshot from video.
    # Within the exp_video_pkl_plot, remember to check the following
    # - path to the video (videoloc)
    # - the time for the snaps (time_snap)
    exp_video_pkl_plot()

if __name__ == '__main__':
    main()