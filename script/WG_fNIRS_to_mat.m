% This MATLAB script can be used to reproduce the result in Figure 12
% Please download BBCItoolbox to 'MyToolboxDir'
% Please download dataset to 'NirsMyDataDir' and 'EegMyDataDir'
% The authors would be grateful if published reports of research using this code 
% (or a modified version, maintaining a significant portion of the original code) would cite the following article:
% Shin et al. "Simultaneous acquisition of EEG and NIRS during cognitive tasks for an open access dataset", 
% Scientific data (2017), under review.

% Modified by Zenghui Wang (scholarzhwang@163.com), May 23, 2026. Paper: A Unified fNIRS Classification Framework Informed by Local Brain Activation Patterns (https://ieeexplore.ieee.org/document/11279989)

clear all; clc; close all;

%%%%%%%%%%%%%%%%%%%%%%%% modify directory paths properly %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
WorkingDir = pwd;
MyToolboxDir = fullfile(WorkingDir,'bbci_public-master');
NirsMyDataDir = fullfile(WorkingDir,'NIRS');
MySaveDir = fullfile(WorkingDir, 'WG_fNIRS_data');
cd(MyToolboxDir);
startup_bbci_toolbox('DataDir',NirsMyDataDir,'TmpDir','/tmp/','History',0);
cd(WorkingDir);


%% initial parameter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subdir_list.nirs = {'VP001-NIRS','VP002-NIRS','VP003-NIRS','VP004-NIRS','VP005-NIRS','VP006-NIRS','VP007-NIRS','VP008-NIRS','VP009-NIRS','VP010-NIRS','VP011-NIRS','VP012-NIRS','VP013-NIRS','VP014-NIRS','VP015-NIRS','VP016-NIRS','VP017-NIRS','VP018-NIRS','VP019-NIRS','VP020-NIRS','VP021-NIRS','VP022-NIRS','VP023-NIRS','VP024-NIRS','VP025-NIRS','VP026-NIRS'};

band_freq = 0.2;
ord = 3;
ival_epo  = [-10 25]*1000; % epoch range (unit: msec)
ival_base = [-5 -2]*1000;  % baseline range (unit: msec)
step_size = 1*1000; % in msec
window_size = 3*1000; % in msec

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for vp = 1 : length(subdir_list.nirs)
    rng(vp);
    disp([subdir_list.nirs{vp}, ' was started']);
    loadDir = fullfile(NirsMyDataDir,subdir_list.nirs{vp});
    cd(loadDir);
    load cnt_wg; load mrk_wg; load mnt_wg;
    cd(WorkingDir);
    %% low-pass filter   
    [z,p,k] = butter(ord, band_freq/cnt_wg.deoxy.fs*2, 'low');
    [SOS,G] = zp2sos(z,p,k);
    
    cnt_wg.deoxy = proc_filtfilt(cnt_wg.deoxy, SOS, G);
    cnt_wg.oxy   = proc_filtfilt(cnt_wg.oxy,   SOS, G);
       
    %% Segmentation
    epo.deoxy = proc_segmentation(cnt_wg.deoxy, mrk_wg, ival_epo);
    epo.oxy   = proc_segmentation(cnt_wg.oxy, mrk_wg, ival_epo);

    %% baseline correction
    epo.deoxy = proc_baseline(epo.deoxy, ival_base);
    epo.oxy = proc_baseline(epo.oxy, ival_base);
    
	%% using moving time windows
    ival_start = (ival_epo(1):step_size:ival_epo(end)-window_size)';
    ival_end = ival_start+window_size;
    ival = [ival_start, ival_end];
    nStep = length(ival);

    for stepIdx = 1:nStep
        segment.deoxy{stepIdx} = proc_selectIval(epo.deoxy, ival(stepIdx,:));
        segment.oxy{stepIdx}   = proc_selectIval(epo.oxy,   ival(stepIdx,:));
    end
    
    % save fNIRS data
    num = num2str(vp);
    mkdir(strcat(MySaveDir, '\',num));
    for stepIdx = 1:nStep
        path = strcat(MySaveDir, '\', num , '\', num2str(stepIdx), '_deoxy.mat');
        signal = segment.deoxy{stepIdx}.x;
        save(path, 'signal');
        
        path = strcat(MySaveDir, '\', num , '\', num2str(stepIdx), '_oxy.mat');
        signal = segment.oxy{stepIdx}.x;
        save(path, 'signal');
    end
    path = strcat(MySaveDir, '\', num , '\', num, '_', 'desc');
    label = segment.deoxy{1}.event.desc; % 1= WG, 2=BL
    save(path, 'label');
    
    disp('WG data finish');
end

display('over')

