clc;
clear;
close all;

addpath( fullfile('matrtklib/MatRTKLIB-main') )

%% Read RINEX observation/navigation file
obss = gt.Gobs(fullfile('E:\TA\rtklib_matlab\data\gnss_log_2024_09_25_15_24_49.24o'));
navs = gt.Gnav(fullfile('E:\TA\rtklib_matlab\data\hksc269h.24*'));

pos = gt.Gpos();
pos.setPos([22.2999014417218,114.177726795579,4.73532988939166], 'llh');

%% Compute residuals (Compensate geometric distance)
ssat = gt.Gsat(obss, navs); % Compute satellite position
ssat.setRcvPos(pos); % Set receiver position
obss = obss.residuals(ssat); % Compute residuals

%% Get total number of epochs
num_epochs = size(obss.L1.P, 1); % 

%% Initialize storage for receiver positions
Receiver_positions = zeros(num_epochs, 3); % 
Receiver_velocities = zeros(num_epochs, 3); % 

%% Initialize cell arrays to store data for each epoch
pseudoranges_meas_cell = cell(num_epochs, 1); % 
satellite_positions_cell = cell(num_epochs, 1); % 
satellite_clock_bias_cell = cell(num_epochs, 1); % 
ionospheric_delay_cell = cell(num_epochs, 1); % 
tropospheric_delay_cell = cell(num_epochs, 1); % 
doppler_meas_cell = cell(num_epochs, 1); % 

%% Loop over all epochs
for epoch = 1:num_epochs
    %% Extract the raw measurement for current epoch
    p_l1 = obss.L1.P(epoch,:);
    sat_prn = obss.sat; % Remove epoch indexing
    sat_pos_x = ssat.x(epoch,:);
    sat_pos_y = ssat.y(epoch,:);
    sat_pos_z = ssat.z(epoch,:);
    sat_pos = [sat_pos_x; sat_pos_y; sat_pos_z];
    sat_clock_err = ssat.dts(epoch,:);
    sat_pos = transpose(sat_pos);
    p_l1 = transpose(p_l1);
    sat_clock_err = transpose(sat_clock_err);
    ion_error_l1 = ssat.ionL1(epoch,:);
    ion_error_l1 = transpose(ion_error_l1);
    tropo_error = ssat.trp(epoch,:);
    tropo_error = transpose(tropo_error);
    sat_sys = ssat.sys; % Remove epoch indexing

    %% Exclude NaN values
    valid_idx = ~isnan(p_l1) & ~isnan(sat_pos(:,1)) & ~isnan(sat_pos(:,2)) & ~isnan(sat_pos(:,3));
    p_l1 = p_l1(valid_idx);
    sat_prn = sat_prn(valid_idx);
    sat_pos = sat_pos(valid_idx,:);
    sat_clock_err = sat_clock_err(valid_idx);
    ion_error_l1 = ion_error_l1(valid_idx);
    tropo_error = tropo_error(valid_idx);
    sat_sys = sat_sys(valid_idx);

    %% Check if enough satellites are available
    if length(p_l1) < 4
        warning(['Epoch ', num2str(epoch), ': Not enough satellites, skipping this epoch.']);
        if epoch > 1
            Receiver_positions(epoch, :) = Receiver_positions(epoch - 1, :); % 保持上一历元的位置
        end
        continue;
    end

    %% Save data into cell arrays
    pseudoranges_meas_cell{epoch} = p_l1;
    satellite_positions_cell{epoch} = sat_pos; % Nx3 matrix
    satellite_clock_bias_cell{epoch} = sat_clock_err;
    ionospheric_delay_cell{epoch} = ion_error_l1;
    tropospheric_delay_cell{epoch} = tropo_error;

    %% Initialize receiver position and clock errors
    if epoch > 1
        Receiver_pos = Receiver_positions(epoch - 1, :)'; % 
    else
        Receiver_pos = [0; 0; 0]; % 
    end
    receiver_clock_err = [0; 0; 0; 0]; % GPS, BEIDOU, GALILEO, GLONASS

    %% Build design matrix and geometric distance
    num_sats = length(p_l1);
    H = zeros(num_sats, 3 + 4); % 
    Geodistance = zeros(num_sats,1);
    for i = 1:num_sats
        Geodistance(i,1) = sqrt((Receiver_pos(1)-sat_pos(i,1))^2 + ...
                                (Receiver_pos(2)-sat_pos(i,2))^2 + ...
                                (Receiver_pos(3)-sat_pos(i,3))^2);
    end
    for i = 1:num_sats
        H(i,1) = (Receiver_pos(1)-sat_pos(i,1))/Geodistance(i,1);
        H(i,2) = (Receiver_pos(2)-sat_pos(i,2))/Geodistance(i,1);
        H(i,3) = (Receiver_pos(3)-sat_pos(i,3))/Geodistance(i,1);
        H(i,4:7) = 0;
        if sat_sys(i) == 1
            H(i,4) = 1; % GPS
        elseif sat_sys(i) == 4
            H(i,5) = 1; % 
        elseif sat_sys(i) == 8
            H(i,6) = 1; % Galileo
        elseif sat_sys(i) == 32
            H(i,7) = 1; % GLONASS
        end
    end

    %% Correct pseudorange measurements
    p_l1_corrected = p_l1 + sat_clock_err - ion_error_l1 - tropo_error;

    %% Least squares estimation
    max_iterations = 10; 
    for iter = 1:max_iterations
        % Calculate the residual
        r = zeros(num_sats,1);
        for i = 1:num_sats
            if sat_sys(i) == 1
                r(i,1) = p_l1_corrected(i) - Geodistance(i,1) - receiver_clock_err(1);
            elseif sat_sys(i) == 4
                r(i,1) = p_l1_corrected(i) - Geodistance(i,1) - receiver_clock_err(2);
            elseif sat_sys(i) == 8
                r(i,1) = p_l1_corrected(i) - Geodistance(i,1) - receiver_clock_err(3);
            elseif sat_sys(i) == 32
                r(i,1) = p_l1_corrected(i) - Geodistance(i,1) - receiver_clock_err(4);
            else
                % 
                r(i,1) = p_l1_corrected(i) - Geodistance(i,1);
            end
        end
        % Calculate the weight matrix
        W = diag(1./(ion_error_l1.^2 + tropo_error.^2 + 1)); % 
        % Calculate the correction
        delta_x = (H'*W*H)\(H'*W*r);
        % Update the receiver position
        Receiver_pos = Receiver_pos + delta_x(1:3);
        receiver_clock_err = receiver_clock_err + delta_x(4:end);
        % Update the design matrix and geometric distance
        for i = 1:num_sats
            Geodistance(i,1) = sqrt((Receiver_pos(1)-sat_pos(i,1))^2 + ...
                                    (Receiver_pos(2)-sat_pos(i,2))^2 + ...
                                    (Receiver_pos(3)-sat_pos(i,3))^2);
        end
        for i = 1:num_sats
            H(i,1) = (Receiver_pos(1)-sat_pos(i,1))/Geodistance(i,1);
            H(i,2) = (Receiver_pos(2)-sat_pos(i,2))/Geodistance(i,1);
            H(i,3) = (Receiver_pos(3)-sat_pos(i,3))/Geodistance(i,1);
            H(i,4:7) = 0;
            if sat_sys(i) == 1
                H(i,4) = 1; % GPS
            elseif sat_sys(i) == 4
                H(i,5) = 1; % 
            elseif sat_sys(i) == 8
                H(i,6) = 1; % Galileo
            elseif sat_sys(i) == 32
                H(i,7) = 1; % GLONASS
            end
        end
        % Check for convergence
        if norm(delta_x) < 1e-4
            break;
        end
    end

    %% Store the receiver position for current epoch
    Receiver_positions(epoch, :) = Receiver_pos';
end

%% After processing all epochs, write the data to CSV files

% Find the maximum number of satellites across all epochs
max_num_sats = max(cellfun(@(x) size(x,1), pseudoranges_meas_cell));

% Initialize matrices with NaNs
pseudoranges_meas_mat = NaN(max_num_sats, num_epochs);
satellite_clock_bias_mat = NaN(max_num_sats, num_epochs);
ionospheric_delay_mat = NaN(max_num_sats, num_epochs);
tropospheric_delay_mat = NaN(max_num_sats, num_epochs);
satellite_positions_mat = NaN(max_num_sats, num_epochs*3); % 3 columns per epoch

% Fill the matrices with data from cell arrays
for epoch = 1:num_epochs
    p_l1 = pseudoranges_meas_cell{epoch};
    sat_clock_err = satellite_clock_bias_cell{epoch};
    ion_error_l1 = ionospheric_delay_cell{epoch};
    tropo_error = tropospheric_delay_cell{epoch};
    sat_pos = satellite_positions_cell{epoch};

    num_sats = length(p_l1);

    pseudoranges_meas_mat(1:num_sats, epoch) = p_l1;
    satellite_clock_bias_mat(1:num_sats, epoch) = sat_clock_err;
    ionospheric_delay_mat(1:num_sats, epoch) = ion_error_l1;
    tropospheric_delay_mat(1:num_sats, epoch) = tropo_error;

    satellite_positions_mat(1:num_sats, (epoch-1)*3+1 : epoch*3) = sat_pos;
end

% Write the matrices to CSV files
writematrix(pseudoranges_meas_mat, 'pseudoranges_meas.csv');
writematrix(satellite_clock_bias_mat, 'satellite_clock_bias.csv');
writematrix(ionospheric_delay_mat, 'ionospheric_delay.csv');
writematrix(tropospheric_delay_mat, 'tropospheric_delay.csv');
writematrix(satellite_positions_mat, 'satellite_positions.csv');

%% Convert ECEF positions to LLH
LLA = ecef2lla(Receiver_positions, 'WGS84'); % [lat, lon, height] in degrees and meters

%% Plot receiver positions on a map
figure;
geoplot(LLA(:,1), LLA(:,2), 'r.-');
geobasemap streets;
title('Receiver Position Trajectory');
