import numpy as np
import math

# Constants
# Minimal distance between the ball and the ball-possessing player (m)
VicinityThreshold = 1.0
# Time required for a player to be considered a ball receiver (in frames)
GracePeriodPlayer = 5
GracePeriodBall = 1
# Minimal ball travel distance of an unsuccessful pass (m)
MinFailedPassLength = 1.5
# Minimal change in trajectory direction considered “significant” (radians)
MinTrChangeAngle = math.pi / 8
# Minimal change in ball speed considered “significant”
MinSpeedChangeFactor = 1.5
# Minimal distance from a goalpost for a kick to be considered a pass (m)
GoalpostDistance = 5.0

# Other constants, that are not described in the paper
# Number of frames to look back to calculate future trajectory of the ball
TrajectoryBack = 3
# Number of frames to generate trajectory for the ball
TrajectoryForward = 6
GoalLength = 7.32
# Borders of the goal area
GoalAreaY = GoalLength / 2 + 5.5
GoalAreaX = 47
# Pitch dimensions
PitchHalfLength = 52.5
PitchHalfWidth = 35  # Should be 34, but there are episodes with larger fields (and the game wasn't stopped)


def is_same_team(p1, p2):
    """
    Check if two players belong to the same team.
    Indexes from 0 to 10 are used for players of the defending team,
    and indexes from 11 to 21 are used for the attacking team.
    """
    return (p1 < 10 and p2 < 10) or (p1 > 10 and p2 > 10)


def is_in_goal_area(object_xy):
    """Checks if the object (ball or player) is in the goal area."""
    return abs(object_xy[0]) > GoalAreaX and abs(object_xy[1]) < GoalAreaY


class EventRecognizer:
    """
    Class for finding events (such as passes and shots) in a particular game episode.
    """

    def __init__(self, episode, show_info=True, show_debug=False):
        self.episode = episode
        self.current_frame = 0
        self.pwb = None
        self.prev_pwb = None
        self.event_frame = None
        self.show_info = show_info
        self.show_debug = show_debug
        self.events = {}

    def find_events(self):
        """
        Main method of the class to analyze the whole episode and compile a list of events.
        :return: dict of events, where key is a frame number
        """
        self.events = {}

        # A main loop
        while self.current_frame < len(self.episode):
            # Find the frame when the ball leaves vicinity of the player
            self.find_ball_faraway()

            # Detecting pass (shot) or possession of a faraway ball
            while self.current_frame < len(self.episode):
                new_event = self.detect_pass_shot()
                self.current_frame += 1
                if new_event is not None:
                    if not (new_event['event'] == 'faraway' or new_event['event'] == 'tackle'):
                        ev_frame = new_event.pop('frame')
                        self.events[str(ev_frame)] = new_event
                    break
        return self.events

    def detect_pass_shot(self):
        """
        Detect shot or pass event, otherwise return None.
        :return: event (dict) or None
        """
        ball_xy = self.get_ball_coordinates(self.current_frame)
        if abs(ball_xy[1]) > PitchHalfWidth:
            self.print_debug('Ball crossed the sideline')
            self.pwb = None
            return self.verify_failed_pass(ball_xy)
        elif abs(ball_xy[0]) > PitchHalfLength:
            self.print_debug('Ball crossed the goal line.')
            self.pwb = None
            if abs(ball_xy[1]) < GoalAreaY and \
                    not (self.prev_pwb == 0 or self.prev_pwb == 11):
                return self.get_event('shot')
            else:
                return self.verify_failed_pass(ball_xy)

        new_pwb = self.get_closest_player(self.current_frame)
        if new_pwb is None:
            return None
        self.print_debug(f'Ball is in vicinity of player {new_pwb}')

        # If the same player controls the ball again
        if new_pwb == self.prev_pwb:
            self.pwb = new_pwb
            return {'frame': int(self.current_frame), 'event': 'faraway', 'player': int(self.pwb)}

        if self.is_possession_changed(new_pwb):
            self.pwb = new_pwb
            if not is_same_team(self.prev_pwb, new_pwb):
                return self.verify_shot(ball_xy)
            return self.get_event('pass', new_pwb)

        return None

    def find_ball_faraway(self):
        """
        Rewind forward to a frame, when the ball is further away than VicinityThreshold from PWB.
        """
        # If there is no pwb, find one
        if self.pwb is None:
            self.find_possession_frame()
        # Find the frame, when ball leaves vicinity of the player
        while self.current_frame < len(self.episode):
            if self.is_ball_faraway(self.pwb, self.current_frame):
                self.prev_pwb = self.pwb
                self.event_frame = self.current_frame - 1
                break
            self.current_frame += 1

    def is_ball_faraway(self, player, frame):
        """
        Check if the ball is further away from the player than
        a VicinityThreshold on a particular frame.
        """
        ball_xy = self.get_ball_coordinates(frame)
        player_xy = self.get_player_coordinates(player, frame)
        dist = np.linalg.norm(player_xy - ball_xy)
        # self.print_debug(f'Distance from player {player} to the ball: {dist}.', frame)
        return dist > VicinityThreshold

    def find_possession_frame(self):
        """
        Rewind forward to a frame where the player controls the ball.
        Set player with a ball (PWB).
        """
        while self.current_frame < len(self.episode):
            pwb_idx = self.get_closest_player(self.current_frame)
            self.current_frame += 1
            if pwb_idx is not None:
                self.pwb = pwb_idx
                break

    def is_possession_changed(self, new_pwb):
        if self.current_frame == len(self.episode) - 1:
            self.print_debug(f'Last frame of the game. Player {new_pwb} is considered to control the ball after.')
            return True

        # Check if speed or direction of the ball has changed
        prev_frame = max(self.current_frame - GracePeriodBall, 0)
        next_frame = min(self.current_frame + GracePeriodBall, len(self.episode) - 1)
        if self.is_ball_speed_changed(prev_frame, next_frame) or \
                self.is_ball_direction_changed(prev_frame, next_frame):
            return True

        # Does new_pwb control the ball from current frame until the furthest frame?
        furthest = min(self.current_frame + GracePeriodPlayer, len(self.episode) - 1)
        for next_frame in range(self.current_frame, furthest + 1):
            if self.is_ball_faraway(new_pwb, next_frame):
                return False
        return True

    def is_ball_speed_changed(self, prev_frame, next_frame):
        prev_speed = self.calculate_ball_speed(prev_frame, self.current_frame)
        next_speed = self.calculate_ball_speed(self.current_frame, next_frame)
        max_speed, min_speed = max(prev_speed, next_speed), min(next_speed, prev_speed)
        if min_speed < 0.001:  # Some small value
            return True  # The ball was still
        return max_speed / min_speed > MinSpeedChangeFactor

    def calculate_ball_speed(self, frame_a, frame_b):
        ball_frame_a = self.get_ball_coordinates(frame_a)
        ball_frame_b = self.get_ball_coordinates(frame_b)
        speed = np.linalg.norm(ball_frame_b - ball_frame_a) / (frame_b - frame_a)
        self.print_debug(f'Ball speed between frames {frame_a} and {frame_b} is {round(speed, 3)}.')
        return speed

    def is_ball_direction_changed(self, prev_frame, next_frame):
        """
        Calculate the angle between ball direction vectors
        """
        ball_prev = self.get_ball_coordinates(prev_frame)
        ball_curr = self.get_ball_coordinates(self.current_frame)
        ball_next = self.get_ball_coordinates(next_frame)
        prev_direction_v = ball_prev - ball_curr
        next_direction_v = ball_curr - ball_next
        angle = np.math.atan2(np.linalg.det([prev_direction_v, next_direction_v]),
                              np.dot(prev_direction_v, next_direction_v))
        self.print_debug(f'Ball trajectory angle between frames ({prev_frame}, {self.current_frame}) and '
                   f'({self.current_frame}, {next_frame}) is {round(np.degrees(angle), 3)} (degrees).')
        return abs(angle) > MinTrChangeAngle

    def get_distances_to_ball(self, frame, ball_xy=None):
        """
        Return an array of distances to the ball for each player.
        If ball_xy is None, get the ball coordinates from the same frame.
        """
        frame_data = self.episode[frame]
        players_xy = frame_data[:-2].reshape(22, 2)
        if ball_xy is None:
            ball_xy = self.get_ball_coordinates(frame)
        # distance to the ball for each player
        ball_dist = np.linalg.norm(players_xy - ball_xy, axis=1)
        return ball_dist

    def get_closest_player(self, frame, in_vicinity_only=True):
        """
        Find an index of a closest player.
        If in_vicinity_only is True, than not further than VicinityThreshold.
        """
        ball_dist = self.get_distances_to_ball(frame)
        closest_idx = np.argmin(ball_dist)
        if in_vicinity_only and ball_dist[closest_idx] > VicinityThreshold:
            return None
        return closest_idx

    def get_closest_teammate(self, frame, ball_xy=None):
        """
        Find an index of a closest teammate.
        Team is determined by index of previous PWB.
        """
        ball_dist = self.get_distances_to_ball(frame, ball_xy)
        # Set distance of prev_pwb to the ball to be a big number,
        # otherwise np.argmin() may return the player himself.
        ball_dist[self.prev_pwb] = 300.0

        if self.prev_pwb < 11:
            closest_tm = np.argmin(ball_dist[:11])
        else:
            closest_tm = np.argmin(ball_dist[11:]) + 11
        self.print_debug(f'Closest teammate idx: {closest_tm}, '
                   f'distance: {round(ball_dist[closest_tm], 3)}.', frame)
        return closest_tm

    def get_ball_coordinates(self, frame):
        """Return x,y coordinates of the ball for a particular frame"""
        return self.episode[frame][-2:]

    def get_player_coordinates(self, player, frame):
        """Return x,y coordinates of the player for a particular frame"""
        player_idx = player * 2
        return self.episode[frame][player_idx:player_idx + 2]

    def verify_failed_pass(self, ball_xy):
        """
        Check if there was a pass attempt.
        :param ball_xy: Coordinates of the ball.
        :return: failed_pass or tackle event (dict)
        """
        # Get coordinates of the PWB on the event_frame
        pwb_xy_prev = self.get_player_coordinates(self.prev_pwb, self.event_frame)
        # Calculate the distance that the ball has travelled
        ball_dist = np.linalg.norm(pwb_xy_prev - ball_xy)
        if ball_dist < MinFailedPassLength:
            self.print_debug('Ball travelled less than MinFailedPassLength.')
            return {'frame': int(self.current_frame), 'event': 'tackle'}

        # TODO: Improve finding of a target player for the failed pass,
        #  look at the ball trajectory instead.
        target_player = self.get_closest_teammate(self.current_frame, ball_xy)
        return self.get_event('failed_pass', target_player)

    def verify_shot(self, ball_xy):
        if is_in_goal_area(ball_xy):
            for next_ball_xy in self.get_ball_trajectory():
                if abs(next_ball_xy[0]) >= PitchHalfLength and \
                        abs(next_ball_xy[1]) < GoalAreaY and \
                        not (self.prev_pwb == 0 or self.prev_pwb == 11):  # not a goalkeeper
                    return self.get_event('shot')
        return self.verify_failed_pass(ball_xy)

    def get_ball_trajectory(self):
        """
        Get an array of ball positions on the pitch for the next TrajectoryForward frames,
        assuming the ball would be traveling with the same speed as during past TrajectoryBack frames.
        :return: list of (x, y) coordinates of the ball
        """
        frame = self.current_frame
        prev_frame = max(0, frame - TrajectoryBack)
        ball_curr = self.get_ball_coordinates(frame)
        ball_prev = self.get_ball_coordinates(prev_frame)
        prev_dir = ball_prev - ball_curr
        coords = []
        for _ in range(TrajectoryForward):
            next_xy = ball_curr - prev_dir / (frame - prev_frame)
            ball_curr = next_xy
            coords.append(next_xy)
            # Break if the ball leaves the pitch
            if abs(next_xy[0]) > PitchHalfLength or abs(next_xy[1]) > PitchHalfWidth:
                break
        return coords

    def get_event(self, event_type, target=None):
        """
        Compile a dictionary object for the event.
        :param event_type: 'shot' or 'pass'
        :param target: target player of the pass
        :return: (dict) event
        """
        event = {'frame': int(self.event_frame), 'event': event_type, 'player': int(self.prev_pwb)}
        if event_type == 'shot':
            self.print_info(f'Frame {self.event_frame}. Shot by player {self.prev_pwb}.')
        else:
            event['target'] = int(target)
            text = 'Pass' if event_type == 'pass' else 'Failed pass'
            self.print_info(f'Frame {self.event_frame}. {text} by player {self.prev_pwb} to {target}.')
        return event

    def print_debug(self, message, frame=None):
        """
        Helper function to control debug messages in one place.
        :param message: text message to print
        :param frame: frame number to print, if None use self.current_frame
        """
        if self.show_debug:
            if frame is None:
                frame = self.current_frame
            print(f'Frame {frame}. {message}')

    def print_info(self, message):
        """
        Helper function to control info messages in one place.
        :param message: text message to print
        """
        if self.show_info:
            print(message)
