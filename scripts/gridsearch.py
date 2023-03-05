from colorama import Fore
from colorama import Style


#def:
#    '''Runs the practice stimulus'''
#    check github linkexists
#    check github linkexists

def run_practice_stimulus():
    '''Runs the practice stimulus'''

    print(f"{Fore.GREEN}{Style.BRIGHT}Starting practice stimulus.{Style.RESET_ALL}")

    print(f"...running practice stimuli")
    # present_audio_and_visual_stimulus(config, is_practice=True)

    print(f"{Fore.GREEN}{Style.BRIGHT}Ending practice session.{Style.RESET_ALL}")

    # Holding until keypress
    input(
        f"{Fore.MAGENTA}{Style.BRIGHT}Please confirm the participant is happy to continue. Press [enter] to continue.{Style.RESET_ALL}")


def run_main_stimulus_loop(number_of_reps, run_number, config):
    '''Runs the main stimulus loop'''

    print(f"{Fore.GREEN}{Style.BRIGHT}Starting repetition loop{Style.RESET_ALL}.")

    for rep in range(run_number, number_of_reps + 1):
        print(f"Ready to start rep {Fore.YELLOW}{Style.BRIGHT}{rep}{Style.RESET_ALL} of {number_of_reps}")

        print(
            f"{Fore.MAGENTA}{Style.BRIGHT}Inform the participant that you are about to start. Request the operator to tell you the HPI coil locations. Request the operator start recording.{Style.RESET_ALL}")
        input(f"{Fore.MAGENTA}{Style.BRIGHT}Press [enter] once you are ready to start the new block.{Style.RESET_ALL}")
        print(f"{Fore.RED}{Style.BRIGHT}If you wish to cancel this run, press [Esc] at any time.{Style.RESET_ALL}")

        print(f"Starting rep {Fore.YELLOW}{Style.BRIGHT}{rep}{Style.RESET_ALL} of {number_of_reps}")
        print(f"...running Stimuli")
        # present_audio_and_visual_stimulus(config)

        print(f"...asking questions (Version {rep})")
        # present_questions(rep, config)

        # Holding until keypress
        print(
            f"{Fore.MAGENTA}{Style.BRIGHT}Inform the participant that that is the end of the {rep} block, and you are saving the data. Inform the operator that you wish to save the block.{Style.RESET_ALL}")
        input(
            f"{Fore.MAGENTA}{Style.BRIGHT}Press [enter] to confirm the participant is happy to continue the next block. (This will not start the next block, instructions for you and the operator will follow).{Style.RESET_ALL}")

    print(f"{Fore.GREEN}{Style.BRIGHT}Ending stimulus.{Style.RESET_ALL}.")


def present_audio_and_visual_stimulus(number_of_seconds: int, data, is_practice: bool):
    '''Runs a single repetition of the stimulus'''

    # Prepare the run
    if is_practice:
        length_in_millisecond = number_of_seconds + 15 * 1000
        # https://psychopy.org/api/sound/playback.html
        auditory_stimulus = sound.Sound('1_practice_run.mp3')
    else:
        length_in_millisecond = 10 * 1000
        auditory_stimulus = sound.Sound('1_practice_run.mp3')

    # Preload the sound in the soundcard ready to play

    # length_in_frames = xyz

    # for this_frame in range (0, length_in_frames):

        # listen for Esc
        # xxxx

        # construct visual stimulus
        # visual_stimulus = create_visual_stimulus(data)

        # confirm time was correct and no missed frames
        # do time check

        # get visual stimulus
        # show_stimuli(visual_stimuli)

        # if this_frame => 5000 and is multiple of 1000:
        # Set make sure 500!!!
        #    set_trigger();

        # set triggers for audio
        # if this time == 1000 and is multiple of 1000:
        #    set_trigger(on time);


        # if this time == 1000 and is multiple of 1000:
            # Set audio stimulus and audiotrigger
            #    wait()
        #now = ptb.GetSecs()
        #mySound.play(when=now + 0.5)
        #    nextFlip = win.getFutureFlipTime(clock='ptb')

            #set_audio_trigger()
            #auditory_stimulus.play(when=now)


        #    win.flip()

        #    set trigger and set_ausio(visual_stimuli)


def create_visual_stimulus(data):
    '''Create visual stimulus'''

    data.top_right_r
    data.top_right_g
    data.top_right_b

    data.top_right_r
    data.top_right_g
    data.top_right_b

    data.top_right_r
    data.top_right_g
    data.top_right_b

    data.top_right_r
    data.top_right_g
    data.top_right_b

    data.top_right_horizontal_pos
    data.top_right_horizontal_pos
    data.top_right_horizontal_pos
    data.top_right_horizontal_pos


def confirm_run(number_of_reps) -> int:
    '''Asks the researcher if the program should start from a specific run.'''

    requested_run_number = input(
        f"{Fore.MAGENTA}{Style.BRIGHT}Do you require a specific run? Press [enter] to start experiment from beginning, otherwise the number followed by [enter]. Waiting for researcher's keypress...{Style.RESET_ALL}")

    if requested_run_number == "":
        requested_run_number = 1
        return requested_run_number

    try:
        requested_run_number_int = int(requested_run_number)
    except ValueError:
        # Handle the exception
        raise

    assert requested_run_number_int < number_of_reps, "Response must be less than or equal than number of reps specified in the config"
    assert requested_run_number_int > 0, "Response must be greater than 0"

    print(f"...starting from rep " + str(requested_run_number_int))
    return requested_run_number_int


