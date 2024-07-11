"""
BNCI 2014-001 Motor imagery dataset.
"""

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.utils import verbose
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
BNCI_URL = "http://bnci-horizon-2020.eu/database/data-sets/"
BBCI_URL = "http://doc.ml.tu-berlin.de/bbci/"


def data_path(url, path=None, force_update=False, update_path=None, verbose=None):
    return [dl.data_dl(url, "BNCI", path, force_update, verbose)]


@verbose
def load_data(
    subject,
    dataset="001-2014",
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    verbose=None,
):  # noqa: D301
    """Get paths to local copies of a BNCI dataset files.

    This will fetch data for a given BNCI dataset. Report to the bnci website
    for a complete description of the experimental setup of each dataset.

    Parameters
    ----------
    subject : int
        The subject to load.
    dataset : string
        The bnci dataset name.
    path : None | str
        Location of where to look for the BNCI data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_BNCI_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the BNCI dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_BNCI_PATH in mne-python
        config to the given path. If None, the user is prompted.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raws : list
        List of raw instances for each non consecutive recording. Depending
        on the dataset it could be a BCI run or a different recording session.
    event_id: dict
        dictonary containing events and their code.
    """
    dataset_list = {
        "001-2014": _load_data_001_2014,
        "004-2014": _load_data_004_2014,
    }

    baseurl_list = {
        "001-2014": BNCI_URL,
        "004-2014": BNCI_URL,
    }

    if dataset not in dataset_list.keys():
        raise ValueError(
            "Dataset '%s' is not a valid BNCI dataset ID. "
            "Valid dataset are %s." % (dataset, ", ".join(dataset_list.keys()))
        )
    return dataset_list[dataset](
        subject, path, force_update, update_path, baseurl_list[dataset], verbose
    )


@verbose
def _load_data_001_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    verbose=None,
):
    """Load data for 001-2014 dataset."""
    if (subject < 1) or (subject > 9):
        raise ValueError("Subject must be between 1 and 9. Got %d." % subject)

    # fmt: off
    ch_names = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
        "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
        "EOG1", "EOG2", "EOG3",
    ]
    # fmt: on
    ch_types = ["eeg"] * 22 + ["eog"] * 3

    sessions = {}
    for r in ["T", "E"]:
        url = "{u}001-2014/A{s:02d}{r}.mat".format(u=base_url, s=subject, r=r)
        filename = data_path(url, path, force_update, update_path)
        runs, ev = _convert_mi(filename[0], ch_names, ch_types)
        # FIXME: deal with run with no event (1:3) and name them
        sessions["session_%s" % r] = {"run_%d" % ii: run for ii, run in enumerate(runs)}
    return sessions


@verbose
def _load_data_004_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    verbose=None,
):
    """Load data for 004-2014 dataset."""
    if (subject < 1) or (subject > 9):
        raise ValueError("Subject must be between 1 and 9. Got %d." % subject)

    ch_names = ["C3", "Cz", "C4", "EOG1", "EOG2", "EOG3"]
    ch_types = ["eeg"] * 3 + ["eog"] * 3

    sessions = []
    for r in ["T", "E"]:
        url = "{u}004-2014/B{s:02d}{r}.mat".format(u=base_url, s=subject, r=r)
        filename = data_path(url, path, force_update, update_path)[0]
        raws, _ = _convert_mi(filename, ch_names, ch_types)
        sessions.extend(raws)

    sessions = {"session_%d" % ii: {"run_0": run} for ii, run in enumerate(sessions)}
    return sessions


def _convert_mi(filename, ch_names, ch_types):
    """
    Processes (Graz) motor imagery data from MAT files, returns list of
    recording runs.
    """
    runs = []
    event_id = {}
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)

    if isinstance(data["data"], np.ndarray):
        run_array = data["data"]
    else:
        run_array = [data["data"]]

    for run in run_array:
        raw, evd = _convert_run(run, ch_names, ch_types, None)
        if raw is None:
            continue
        runs.append(raw)
        event_id.update(evd)
    # change labels to match rest
    standardize_keys(event_id)
    return runs, event_id


def standardize_keys(d):
    master_list = [
        ["both feet", "feet"],
        ["left hand", "left_hand"],
        ["right hand", "right_hand"],
        ["FEET", "feet"],
        ["HAND", "right_hand"],
        ["NAV", "navigation"],
        ["SUB", "subtraction"],
        ["WORD", "word_ass"],
    ]
    for old, new in master_list:
        if old in d.keys():
            d[new] = d.pop(old)


@verbose
def _convert_run(run, ch_names=None, ch_types=None, verbose=None):
    """Convert one run to raw."""
    # parse eeg data
    event_id = {}
    n_chan = run.X.shape[1]
    montage = make_standard_montage("standard_1005")
    eeg_data = 1e-6 * run.X
    sfreq = run.fs

    if not ch_names:
        ch_names = ["EEG%d" % ch for ch in range(1, n_chan + 1)]
        montage = None  # no montage

    if not ch_types:
        ch_types = ["eeg"] * n_chan

    trigger = np.zeros((len(eeg_data), 1))
    # some runs does not contains trials i.e baseline runs
    if len(run.trial) > 0:
        trigger[run.trial - 1, 0] = run.y
    else:
        return None, None

    eeg_data = np.c_[eeg_data, trigger]
    ch_names = ch_names + ["stim"]
    ch_types = ch_types + ["stim"]
    event_id = {ev: (ii + 1) for ii, ev in enumerate(run.classes)}
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    raw.set_montage(montage)
    return raw, event_id


class MNEBNCI(BaseDataset):
    """Base BNCI dataset"""

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        sessions = load_data(subject=subject, dataset=self.code, verbose=False)
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return load_data(
            subject=subject,
            dataset=self.code,
            verbose=verbose,
            update_path=update_path,
            path=path,
            force_update=force_update,
        )


class BNCI2014001(MNEBNCI):
    """BNCI 2014-001 Motor Imagery dataset.

    .. admonition:: Dataset summary


        ===========  =======  =======  ==========  =================  ============  ===============  ===========
        Name           #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        ===========  =======  =======  ==========  =================  ============  ===============  ===========
        BNCI2014001       10       22           4                144  4s            250Hz                      2
        ===========  =======  =======  ==========  =================  ============  ===============  ===========

    Dataset IIa from BCI Competition 4 [1]_.

    **Dataset Description**

    This data set consists of EEG data from 9 subjects.  The cue-based BCI
    paradigm consisted of four different motor imagery tasks, namely the imag-
    ination of movement of the left hand (class 1), right hand (class 2), both
    feet (class 3), and tongue (class 4).  Two sessions on different days were
    recorded for each subject.  Each session is comprised of 6 runs separated
    by short breaks.  One run consists of 48 trials (12 for each of the four
    possible classes), yielding a total of 288 trials per session.

    The subjects were sitting in a comfortable armchair in front of a computer
    screen.  At the beginning of a trial ( t = 0 s), a fixation cross appeared
    on the black screen.  In addition, a short acoustic warning tone was
    presented.  After two seconds ( t = 2 s), a cue in the form of an arrow
    pointing either to the left, right, down or up (corresponding to one of the
    four classes left hand, right hand, foot or tongue) appeared and stayed on
    the screen for 1.25 s.  This prompted the subjects to perform the desired
    motor imagery task.  No feedback was provided.  The subjects were ask to
    carry out the motor imagery task until the fixation cross disappeared from
    the screen at t = 6 s.

    Twenty-two Ag/AgCl electrodes (with inter-electrode distances of 3.5 cm)
    were used to record the EEG; the montage is shown in Figure 3 left.  All
    signals were recorded monopolarly with the left mastoid serving as
    reference and the right mastoid as ground. The signals were sampled with.
    250 Hz and bandpass-filtered between 0.5 Hz and 100 Hz. The sensitivity of
    the amplifier was set to 100 μV . An additional 50 Hz notch filter was
    enabled to suppress line noise

    References
    ----------

    .. [1] Tangermann, M., Müller, K.R., Aertsen, A., Birbaumer, N., Braun, C.,
           Brunner, C., Leeb, R., Mehring, C., Miller, K.J., Mueller-Putz, G.
           and Nolte, G., 2012. Review of the BCI competition IV.
           Frontiers in neuroscience, 6, p.55.
    """
    def __init__(self, signal_type):
        if signal_type == "MI":
            events = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4},
            interval = [3, 5.995]
        elif signal_type == "REST":
            events = {"rest": 5}
            interval = [0, 2.995]
        else:
            raise ValueError("signal_type must be 'MI' or 'REST'")
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=2,
            events=events,
            code="001-2014",
            interval=interval,
            paradigm="imagery",
            doi="10.3389/fnins.2012.00055",
        )


class BNCI2014002(MNEBNCI):
    """BNCI 2014-002 Motor Imagery dataset.

    .. admonition:: Dataset summary


        ===========  =======  =======  ==========  =================  ============  ===============  ===========
        Name           #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        ===========  =======  =======  ==========  =================  ============  ===============  ===========
        BNCI2014002       15       15           2                 80  5s            512Hz                      1
        ===========  =======  =======  ==========  =================  ============  ===============  ===========

    Motor Imagery Dataset from [1]_.

    **Dataset description**

    The session consisted of eight runs, five of them for training and three
    with feedback for validation.  One run was composed of 20 trials.  Taken
    together, we recorded 50 trials per class for training and 30 trials per
    class for validation.  Participants had the task of performing sustained (5
    seconds) kinaesthetic motor imagery (MI) of the right hand and of the feet
    each as instructed by the cue. At 0 s, a white colored cross appeared on
    screen, 2 s later a beep sounded to catch the participant’s attention. The
    cue was displayed from 3 s to 4 s. Participants were instructed to start
    with MI as soon as they recognized the cue and to perform the indicated MI
    until the cross disappeared at 8 s. A rest period with a random length
    between 2 s and 3 s was presented between trials. Participants did not
    receive feedback during training.  Feedback was presented in form of a
    white
    coloured bar-graph.  The length of the bar-graph reflected the amount of
    correct classifications over the last second.  EEG was measured with a
    biosignal amplifier and active Ag/AgCl electrodes (g.USBamp, g.LADYbird,
    Guger Technologies OG, Schiedlberg, Austria) at a sampling rate of 512 Hz.
    The electrodes placement was designed for obtaining three Laplacian
    derivations.  Center electrodes at positions C3, Cz, and C4 and four
    additional electrodes around each center electrode with a distance of 2.5
    cm, 15 electrodes total.  The reference electrode was mounted on the left
    mastoid and the ground electrode on the right mastoid.  The 13 participants
    were aged between 20 and 30 years, 8 naive to the task, and had no known
    medical or neurological diseases.

    References
    -----------

    .. [1] Steyrl, D., Scherer, R., Faller, J. and Müller-Putz, G.R., 2016.
           Random forests in non-invasive sensorimotor rhythm brain-computer
           interfaces: a practical and convenient non-linear classifier.
           Biomedical Engineering/Biomedizinische Technik, 61(1), pp.77-86.

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 15)),
            sessions_per_subject=1,
            events={"right_hand": 1, "feet": 2},
            code="002-2014",
            interval=[3, 8],
            paradigm="imagery",
            doi="10.1515/bmt-2014-0117",
        )


class BNCI2014004(MNEBNCI):
    """BNCI 2014-004 Motor Imagery dataset.

    .. admonition:: Dataset summary


        ===========  =======  =======  ==========  =================  ============  ===============  ===========
        Name           #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        ===========  =======  =======  ==========  =================  ============  ===============  ===========
        BNCI2014004       10        3           2                360  4.5s          250Hz                      5
        ===========  =======  =======  ==========  =================  ============  ===============  ===========

    Dataset B from BCI Competition 2008.

    **Dataset description**

    This data set consists of EEG data from 9 subjects of a study published in
    [1]_. The subjects were right-handed, had normal or corrected-to-normal
    vision and were paid for participating in the experiments.
    All volunteers were sitting in an armchair, watching a flat screen monitor
    placed approximately 1 m away at eye level. For each subject 5 sessions
    are provided, whereby the first two sessions contain training data without
    feedback (screening), and the last three sessions were recorded with
    feedback.

    Three bipolar recordings (C3, Cz, and C4) were recorded with a sampling
    frequency of 250 Hz.They were bandpass- filtered between 0.5 Hz and 100 Hz,
    and a notch filter at 50 Hz was enabled.  The placement of the three
    bipolar recordings (large or small distances, more anterior or posterior)
    were slightly different for each subject (for more details see [1]).
    The electrode position Fz served as EEG ground. In addition to the EEG
    channels, the electrooculogram (EOG) was recorded with three monopolar
    electrodes.

    The cue-based screening paradigm consisted of two classes,
    namely the motor imagery (MI) of left hand (class 1) and right hand
    (class 2).
    Each subject participated in two screening sessions without feedback
    recorded on two different days within two weeks.
    Each session consisted of six runs with ten trials each and two classes of
    imagery.  This resulted in 20 trials per run and 120 trials per session.
    Data of 120 repetitions of each MI class were available for each person in
    total.  Prior to the first motor im- agery training the subject executed
    and imagined different movements for each body part and selected the one
    which they could imagine best (e. g., squeezing a ball or pulling a brake).

    Each trial started with a fixation cross and an additional short acoustic
    warning tone (1 kHz, 70 ms).  Some seconds later a visual cue was presented
    for 1.25 seconds.  Afterwards the subjects had to imagine the corresponding
    hand movement over a period of 4 seconds.  Each trial was followed by a
    short break of at least 1.5 seconds.  A randomized time of up to 1 second
    was added to the break to avoid adaptation

    For the three online feedback sessions four runs with smiley feedback
    were recorded, whereby each run consisted of twenty trials for each type of
    motor imagery.  At the beginning of each trial (second 0) the feedback (a
    gray smiley) was centered on the screen.  At second 2, a short warning beep
    (1 kHz, 70 ms) was given. The cue was presented from second 3 to 7.5. At
    second 7.5 the screen went blank and a random interval between 1.0 and 2.0
    seconds was added to the trial.

    References
    ----------

    .. [1] R. Leeb, F. Lee, C. Keinrath, R. Scherer, H. Bischof,
           G. Pfurtscheller. Brain-computer communication: motivation, aim,
           and impact of exploring a virtual apartment. IEEE Transactions on
           Neural Systems and Rehabilitation Engineering 15, 473–482, 2007

    """

    def __init__(self, signal_type):
        if signal_type == "MI":
            events = {"left_hand": 1, "right_hand": 2}
            interval = [3, 5.995]
        elif signal_type == "REST":
            events = {"rest": 5}
            interval = [0, 2.995]
        else:
            raise ValueError("signal_type must be 'MI' or 'REST'")
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=5,
            events=events,
            code="004-2014",
            interval=interval,
            paradigm="imagery",
            doi="10.1109/TNSRE.2007.906956",
        )
