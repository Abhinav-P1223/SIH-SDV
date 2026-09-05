function replay_telemetry(logPath, repoRoot)
% replay_telemetry - Plot and check a JSONL telemetry log written by JsonLinesSink.
%
%   replay_telemetry()                                  % ../logs/sudden_cattle_crossing.jsonl
%   replay_telemetry('../logs/mixed_traffic_curve.jsonl')
%   replay_telemetry(logPath, repoRoot)                 % repoRoot holds config/vehicle.yaml
%
% UNTESTED IN MATLAB: written without a MATLAB installation. It uses only base
% MATLAB (fgetl, jsondecode, regexp, plot, assert). Please report syntax
% slips against the JSONL layout documented in autonomy/telemetry/telemetry.py.
%
% Log layout (one JSON object per line, TelemetryFrame.to_dict()):
%   timestamp, step, planning_cycle, ego{x,y,yaw,longitudinal_velocity,
%   longitudinal_acceleration,steering_angle,...}, control{steering_angle,
%   acceleration,brake,source}, risk{max_level,max_score,min_ttc,...},
%   behavior{state,reason,...} or null, safety{override_active,...},
%   metrics{minimum_obstacle_clearance,collision_count,scenario_completed,...}.
% Planning-cycle frames additionally carry plan{...}, predictions, road.
% Python inf is written as null, which jsondecode returns as [] -> NaN here.
%
% Invariants checked (the same ones tests/scenarios/*.py assert in Python):
%   collision_count == 0, scenario completed, min clearance >= planning.safety_margin_m,
%   |steering| <= max_steering_angle, |d steering / dt| <= max_steering_rate,
%   -max_deceleration <= acceleration <= max_acceleration, speed <= max_speed.

here = fileparts(mfilename('fullpath'));
if nargin < 2 || isempty(repoRoot), repoRoot = fullfile(here, '..'); end
if nargin < 1 || isempty(logPath), logPath = fullfile(repoRoot, 'logs', 'sudden_cattle_crossing.jsonl'); end

veh = read_two_level_yaml(fullfile(repoRoot, 'config', 'vehicle.yaml'));
auto = read_two_level_yaml(fullfile(repoRoot, 'config', 'autonomy.yaml'));
maxSteer = deg2rad(veh.vehicle.max_steering_angle_deg);
maxSteerRate = deg2rad(veh.vehicle.max_steering_rate_deg_s);
maxAccel = veh.vehicle.max_acceleration_mps2;
maxDecel = veh.vehicle.max_deceleration_mps2;
maxSpeed = veh.vehicle.max_speed_mps;
safetyMargin = auto.planning.safety_margin_m;

% ---- load frames -----------------------------------------------------------
fid = fopen(logPath, 'r');
assert(fid > 0, 'cannot open %s', logPath);
cleaner = onCleanup(@() fclose(fid));
t = []; v = []; steer = []; accel = []; state = {}; override = []; clearance = []; ttc = []; planClear = [];
collisions = 0; completed = false; termination = '';
line = fgetl(fid);
while ischar(line)
    if ~isempty(strtrim(line))
        f = jsondecode(line);
        t(end+1) = f.timestamp;                                          %#ok<AGROW>
        v(end+1) = f.ego.longitudinal_velocity;                          %#ok<AGROW>
        steer(end+1) = f.ego.steering_angle;                             %#ok<AGROW>
        accel(end+1) = f.ego.longitudinal_acceleration;                  %#ok<AGROW>
        if isstruct(f.behavior), state{end+1} = f.behavior.state; else, state{end+1} = 'NONE'; end %#ok<AGROW>
        override(end+1) = logical(f.safety.override_active);             %#ok<AGROW>
        clearance(end+1) = num_or_nan(f.metrics.minimum_obstacle_clearance); %#ok<AGROW>
        ttc(end+1) = num_or_nan(f.risk.min_ttc);                          %#ok<AGROW>
        if isfield(f, 'plan') && isstruct(f.plan) && isfield(f.plan, 'selected') && isstruct(f.plan.selected)
            planClear(end+1) = num_or_nan(f.plan.selected.min_clearance); %#ok<AGROW>
        else
            planClear(end+1) = NaN;                                      %#ok<AGROW>
        end
        collisions = f.metrics.collision_count;
        completed = logical(f.metrics.scenario_completed);
        termination = f.metrics.termination_reason;
    end
    line = fgetl(fid);
end
assert(numel(t) > 1, 'no frames in %s', logPath);
dt = median(diff(t));
fprintf('%d frames, %.2f s, dt = %.3f s, termination = %s\n', numel(t), t(end) - t(1), dt, termination);

% ---- plots -----------------------------------------------------------------
[stateNames, ~, stateIdx] = unique(state, 'stable');
figure('Name', ['telemetry replay: ' logPath], 'Color', 'w');
ax1 = subplot(4, 1, 1); plot(t, v, 'LineWidth', 1.2); hold on; yline(maxSpeed, 'r--');
ylabel('speed [m/s]'); grid on; title(strrep(logPath, '\', '/'), 'Interpreter', 'none');
shade_override(t, override);
ax2 = subplot(4, 1, 2); plot(t, rad2deg(steer), 'LineWidth', 1.2); hold on;
yline(rad2deg(maxSteer), 'r--'); yline(-rad2deg(maxSteer), 'r--'); ylabel('steering [deg]'); grid on;
ax3 = subplot(4, 1, 3); stairs(t, stateIdx, 'LineWidth', 1.4); grid on;
set(ax3, 'YTick', 1:numel(stateNames), 'YTickLabel', stateNames, 'YLim', [0.5 numel(stateNames) + 0.5]);
ylabel('behaviour state');
ax4 = subplot(4, 1, 4); plot(t, clearance, 'LineWidth', 1.2); hold on; plot(t, planClear, '.');
yline(safetyMargin, 'r--'); ylabel('min clearance [m]'); xlabel('time [s]'); grid on;
legend({'executed (running min)', 'selected plan', 'safety margin'}, 'Location', 'best');
linkaxes([ax1 ax2 ax3 ax4], 'x');

% ---- invariants ------------------------------------------------------------
tol = 1e-6;
assert(collisions == 0, 'collision_count = %d', collisions);
assert(completed, 'scenario not completed (%s)', termination);
assert(all(isnan(clearance) | clearance >= safetyMargin - tol), 'min clearance %.3f < safety margin %.3f', min(clearance), safetyMargin);
assert(all(abs(steer) <= maxSteer + tol), 'steering angle limit violated');
rate = abs(diff(steer)) ./ diff(t);
assert(all(rate <= maxSteerRate + 1e-3), 'steering rate limit violated: %.3f > %.3f rad/s', max(rate), maxSteerRate);
assert(all(accel <= maxAccel + tol) && all(accel >= -maxDecel - tol), 'acceleration limits violated');
assert(all(v <= maxSpeed + tol), 'speed limit violated');
fprintf('REPLAY CHECKS PASSED: no collision, completed, clearance >= %.2f m, actuator limits respected.\n', safetyMargin);
fprintf('min TTC seen: %.2f s\n', min(ttc, [], 'omitnan'));
end

% ---------------------------------------------------------------------------
function x = num_or_nan(x)
% Python inf / None arrive as null -> [] from jsondecode.
if isempty(x), x = NaN; end
end

function shade_override(t, override)
% Shade intervals where the safety supervisor overrode the controller.
edges = diff([false override false]);
starts = find(edges == 1); stops = find(edges == -1) - 1;
yl = ylim;
for k = 1:numel(starts)
    patch([t(starts(k)) t(stops(k)) t(stops(k)) t(starts(k))], [yl(1) yl(1) yl(2) yl(2)], ...
          [1 0.6 0.6], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
end
end

function cfg = read_two_level_yaml(path)
% Minimal reader for the flat "section: / key: value  # comment" layout of config/*.yaml.
% Not a general YAML parser: it handles top-level sections and one level of scalar keys.
cfg = struct();
section = '';
fid = fopen(path, 'r');
assert(fid > 0, 'cannot open %s', path);
cleaner = onCleanup(@() fclose(fid));
line = fgetl(fid);
while ischar(line)
    code = regexprep(line, '#.*$', '');
    top = regexp(code, '^([A-Za-z_][A-Za-z0-9_]*):\s*$', 'tokens', 'once');
    kv = regexp(code, '^\s+([A-Za-z_][A-Za-z0-9_]*):\s*(\S.*?)\s*$', 'tokens', 'once');
    if ~isempty(top)
        section = top{1};
        if ~isfield(cfg, section), cfg.(section) = struct(); end
    elseif ~isempty(kv) && ~isempty(section)
        value = str2double(kv{2});
        if isnan(value)
            value = kv{2};
            if strcmpi(value, 'true'), value = true; elseif strcmpi(value, 'false'), value = false; end
        end
        cfg.(section).(kv{1}) = value;
    end
    line = fgetl(fid);
end
end
