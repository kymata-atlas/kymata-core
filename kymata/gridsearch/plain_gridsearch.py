from colorama import Fore
from colorama import Style


def do_gridsearch_on_both_hemsipheres(neurophysiology_data_file_directory,
                    predicted_function_outputs_data,
                    hexel_expression_master,
                    functions_to_apply_gridsearch) -> XYZ:
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

    #Check if the master expression already contains it

    #Print out similar, and check.

    for hemi in ['left', 'right']:

        print(f"...for {hemi}")

        do_gridsearch()


    print(f"{Fore.GREEN}{Style.BRIGHT}Ending Gridsearch. Saving in master expression file. {Style.RESET_ALL}")

    return xx

def do_gridsearch():
    '''Do the gridsearch'''

    print(f"   ...XYZ")

    dentritic current_data_array = new array()
    prediction_waveform = load_data.load_prediction_waveform()


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

        # Transform populations with fisher-z

        # First eliviate rounding of - 0.999999 causing problems with log() in fisher-z transform.
        allcorrs(allcorrs < -0.999999) = -0.999999;
        allcorrs(allcorrs > 0.999999) = 0.999999;

        # Transform fisher-z
        allcorrs = 0.5 * log((1 + allcorrs). / (1 - allcorrs));

        truewordsCorr = reshape(allcorrs(:, 1,:), nVertices, nWords, 1);
        randwordsCorr = reshape(allcorrs(:, 2: end,:), nVertices, (nWords * numberOfIterations) - nWords, 1);

        if debug:
                # lillefor test for Guassianism
                vertexnumber = 2444;
                h = lillietest(randwordsCorr(vertexnumber,:));
                histfit(randwordsCorr(vertexnumber,:), 40); # plot
                histfit(truewordsCorr(vertexnumber,:), 40);
                h = findobj(gca, 'Type', 'patch');
                display(h);
                set(h(1), 'FaceColor', [0.982 0.909 0.721], 'EdgeColor', 'k');
                set(h(2), 'FaceColor', [0.819 0.565 0.438], 'EdgeColor', 'k');
                And do the little graph disribution box plots below as well.

        # Do population test on signals

        pvalues = zeros(1, nVertices);
        for vertex = 1:nVertices:
            truepopulation = truewordsCorr(vertex,:);
            randpopulation = randwordsCorr(vertex,:);

            # 2-sample t-test

            [h, p, ci, stats] = ttest2(truepopulation, randpopulation,
                                       1 - ((1 - f_alpha) ^ (1 / (2 * length(latencies) * nVertices))), 'both', 'unequal');
            pvalues(1, vertex) = p

            # Save at correct latency in STC
            outputSTC.data((latency) + (pre_stimulus_window + 1),:) = pvalues

            clear MEGdata wordsignalstrueDownsampled MEGdataDownsampled R  randwordsGuassian truewordsGuassian  truewords randwords;



def do_ad_hoc_power_calculation_on_XYZ(n_samples, alpha, power_val)
    'This works out the power needed on ZYX'

    n_samples = 15000
    h number of samples per group
    alpha = 0.0000005  # significance level
    power_val = 0.8  # desired power
    standard_deviation = xyz
    ratio = n_samples1/n_samples2

    # calculate the minimum detectable effect size
    meff = power.tt_ind_solve_power(alpha=alpha, power=power_val, nobs1=n_samples, ratio=ratio, alternative='larger', sd = sd)

    print(f"Minimum detectable effect size: {meff:.3f}")