from colorama import Fore
from colorama import Style


def do_gridsearch():
    '''Do the Kymata gridsearch over all hexels for all latencies'''

    print(f"{Fore.GREEN}{Style.BRIGHT}Starting gridsearch{Style.RESET_ALL}")

    # Load variables
    functionname = 'TVL2002 Overall instantaneous loudness'
    functionname = 'xxxx'
    inputstream = 'audio';

    experimentName = ['DATASET_3-01_visual-and-auditory'];
    itemlistFilename = [rootDataSetPath, experimentName, '/items.txt'];

    pre_stimulus_window = 200 # in milliseconds
    post_stimulus_window = 800 # in milliseconds

    latency_step = 5 # in milliseconds
    latencies = [-200:latency_step:800]
    number_of_iterations = 5
    downsample_rate = 4
    cuttoff = 1000 # in milliseconds

    print(f"...for Function: {xxxxx}")

        Check if the master expression already contains it

        Print out similar, and check.

        for hemi in ['left', 'right']:

            print(f"...for {hemi}")

            # Polarity represents a special case of the function search - that
            # where the waveform is flipped in it's polarity. The default action
            # in Kymata is to check both, and then select the best, as we don't
            # care about the polarity.
            for polaritiy in ['positive', 'negitive']

                print(f"   ...for {polaritiy}")

                do gridsearch_for_polarity()

            print(f"   ...merging polarities")

            merge polarities



    print(f"{Fore.GREEN}{Style.BRIGHT}Ending Gridsearch. Saving in master expression file. {Style.RESET_ALL}")

    return xx


def do_gridsearch_for_polarity(polarity:String) -> STC file:
    '''Do the gridsearch for a given polarity'''

    print(f"   ...XYZ")

    dentritic current_data_array = new array()
    prediction_waveform = load_data.load_prediction_waveform()

    if polarity = 'negative'
        prediction_waveform = prediction_waveform * -1

    vecotrise and use Numba/CyPY/CUDA or NUMBa and ...
    # cut to trial length
    wordsignalstrue = wordsignalstrue(:, 1: cutoff)';
    wordsignalstrue = repmat(wordsignalstrue, [1 1 nVertices]);
    wordsignalstrue = permute(wordsignalstrue, [3 1 2]);
    wordsignalstrue = single(wordsignalstrue);
    % wordsignalstrue(wordsignalstrue == 0) = nan;

    # create permorder
    [junk permorder] = sort(rand(400, nWords), 2);
    permorder = unique([1:nWords; permorder], 'rows');
    permorder(any(bsxfun( @ minus, permorder, 1: nWords) == 0, 2),:) = []; permorder = [1:nWords; permorder];
    permorder = permorder(1:numberOfIterations + 1,:);

    # Convert MEG - signals true, in same order as wordlist

    allMEGdata = single(zeros(nVertices, nTimePoints, nWords));

    for w in 1:numel(wordlist):
        thisword = char(wordlist(w));

        inputfilename = [rootCodeOutputPath version '/' experimentName, '/5-averaged-by-trial-data/', inputfolder, '/', thisword
                     '-' leftright '.stc'];

        disp(num2str(w));

        sensordata = mne_read_stc_file(inputfilename);
        allMEGdata(:,:, w) = sensordata.data(:,:)

    clear('-regexp', '^grand');
    clear other as well

    # Start Matching / Mismatching proceedure for each time point
    for latency in latencies:

        print(f"   ...Latency ', num2str(q) ' out of ', num2str(length(latencies))]")

        MEGdata = allMEGdata(:, (pre_stimulus_window + latency + 1): (pre_stimulus_window + cutoff + latency),:);

        # downsample
        MEGdataDownsampled = MEGdata(:, 1: downsample_rate:end,:);
        clear MEGdata;
        wordsignalstrueDownsampled = wordsignalstrue(:, 1: downsample_rate:end,:);

        if debug:
            scatter(MEGdata(1,:, 1), wordsignalstrue(1,:, 1));
            scatter(MEGdataDownsampled(1,:, 1), wordsignalstrueDownsampled(1,:, 1));

        # -------------------------------
        # Matched / Mismatched
        # -------------------------------

        allcorrs = zeros(nVertices, numberOfIterations, nWords);

        for i in 1:numberOfIterations:

            disp(['Iteration ', num2str(i) ' out of ', num2str(numberOfIterations)]);

            shuffledMEGdata = permute(MEGdataDownsampled, [3 2 1])
            downsampledtimespan = size(shuffledMEGdata, 2)
            shuffledMEGdata = reshape(shuffledMEGdata, nWords, nVertices * downsampledtimespan)
            shuffledMEGdata = shuffledMEGdata(permorder(i,:),:)
            shuffledMEGdata = reshape(shuffledMEGdata, nWords, downsampledtimespan, nVertices)
            shuffledMEGdata = permute(shuffledMEGdata, [3 2 1])

            # Confirm signals are not shorter than cutt-off
            assert XYZ

            # Do  correlation
            allcorrs(:, i,:) = nansum(bsxfun( @ minus, shuffledMEGdata,
                                      nanmean(shuffledMEGdata, 2)). * bsxfun( @ minus, wordsignalstrueDownsampled, nanmean(
            wordsignalstrueDownsampled, 2)), 2)./ (sqrt(
            nansum((bsxfun( @ minus, shuffledMEGdata, nanmean(shuffledMEGdata, 2))). ^ 2, 2)). * sqrt(
            nansum((bsxfun( @ minus, wordsignalstrueDownsampled, nanmean(wordsignalstrueDownsampled, 2))). ^ 2, 2)));


        clear shuffledMEGdata;

        % -------------------------------
        % Transform populations with fisher - z
        % -------------------------------

        # eliviates rounding of - 0.999999 causing problems with log() in fisher-z transform.
        allcorrs(allcorrs < -0.999999) = -0.999999;
        allcorrs(allcorrs > 0.999999) = 0.999999;

        allcorrs = 0.5 * log((1 + allcorrs). / (1 - allcorrs));

        truewordsCorr = reshape(allcorrs(:, 1,:), nVertices, nWords, 1);
        randwordsCorr = reshape(allcorrs(:, 2: end,:), nVertices, (nWords * numberOfIterations) - nWords, 1);

        if debug:
                # lillefor test for Guassianism
                vertexnumber = 2444;
                h = lillietest(randwordsCorr(vertexnumber,:));
                histfit(randwordsCorr(vertexnumber,:), 40); % plot
                histfit(truewordsCorr(vertexnumber,:), 40);
                h = findobj(gca, 'Type', 'patch');
                display(h);
                set(h(1), 'FaceColor', [0.982 0.909 0.721], 'EdgeColor', 'k');
                set(h(2), 'FaceColor', [0.819 0.565 0.438], 'EdgeColor', 'k');
                And do the little graph disribution box plots below as well.

        # Do population test on signals

        pvalues = zeros(1, nVertices);
        for vertex = 1:nVertices
            truepopulation = truewordsCorr(vertex,:);
            randpopulation = randwordsCorr(vertex,:);

            # 2-sample t-test

            [h, p, ci, stats] = ttest2(truepopulation, randpopulation,
                                       1 - ((1 - f_alpha) ^ (1 / (2 * length(latencies) * nVertices))), 'right', 'unequal');
            pvalues(1, vertex) = p

            # Save at correct latency in STC
            outputSTC.data((latency) + (pre_stimulus_window + 1),:) = pvalues

            clear MEGdata wordsignalstrueDownsampled MEGdataDownsampled R  randwordsGuassian truewordsGuassian  truewords randwords;

