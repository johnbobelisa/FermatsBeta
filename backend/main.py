import heapq
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, NamedTuple, Any, Literal
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class Point(NamedTuple):
	"""
	NamedTuple instances are immutable and very low overhead. 
	"""
	x: float
	y: float

@dataclass
class Hold:
	""" Represents a single hold on the wall. """
	id: str # 'h1', 'h2', etc.
	type: str # 'start', 'regular', 'finish'
	pixel_position: Point # 'position in original pixel coordinates'
	real_position: Point # 'position in real-world coordinates (cm)'

@dataclass
class Climber:
	""" Represents the climber's anthropometry with fixed pixel lengths. """
	height_cm: float
	arm_span_cm: float
	scale_factor: float = field(init=False)  # Will be set by input Problem JSON

	# Segment lengths (cm)
	upper_arm_cm: float = field(init=False)
	forearm_cm: float = field(init=False)
	thigh_cm: float = field(init=False)
	shin_cm: float = field(init=False)

	# Segment lengths (px)
	upper_arm_px: float = field(init=False)
	forearm_px: float = field(init=False)
	thigh_px: float = field(init=False)
	shin_px: float = field(init=False)

	# Maximum reach (cm & px)
	max_arm_reach_cm: float = field(init=False)
	max_leg_reach_cm: float = field(init=False)
	max_arm_reach_px: float = field(init=False)
	max_leg_reach_px: float = field(init=False)

	def cm_to_px(self, cm: float) -> float:
		"""Converts centimeters to pixels using the scale factor."""
		return cm / self.scale_factor
	
	def set_scale_and_calculate(self, scale_factor: float) -> None:
		""" Sets the image scale factor and calculates all anthropometric dimensions. """
		self.scale_factor = scale_factor

		# --- Calculate segment lengths in cm ---
		self.upper_arm_cm = self.arm_span_cm * 0.22
		self.forearm_cm = self.arm_span_cm * 0.22
		self.thigh_cm = self.height_cm * 0.24
		self.shin_cm = self.height_cm * 0.23

		# --- Convert to pixels ---
		self.upper_arm_px = self.cm_to_px(self.upper_arm_cm)
		self.forearm_px = self.cm_to_px(self.forearm_cm)
		self.thigh_px = self.cm_to_px(self.thigh_cm)
		self.shin_px = self.cm_to_px(self.shin_cm)

		# --- Compute maximum reach ---
		self.max_arm_reach_cm = self.upper_arm_cm + self.forearm_cm
		self.max_leg_reach_cm = self.thigh_cm + self.shin_cm
		self.max_arm_reach_px = self.upper_arm_px + self.forearm_px
		self.max_leg_reach_px = self.thigh_px + self.shin_px

@dataclass(frozen=True)
class State:
	"""
	Represents the climber's position as a tuple of hold IDs.
	frozen=True makes State immutable.
	"""
	lh_id: Optional[str]  # Left Hand
	rh_id: Optional[str]  # Right Hand
	lf_id: Optional[str]  # Left Foot
	rf_id: Optional[str]  # Right Foot

@dataclass
class Problem:
	"""The main container for all problem-specific data."""
	problem_name: str
	climber: Climber
	holds: Dict[str, Hold]
	scale_factor: float
	start_holds: List[str] = field(init=False)
	finish_holds: List[str] = field(init=False)

	def __post_init__(self):
		"""Separate holds by type for easy access."""
		self.start_holds = [h.id for h in self.holds.values() if h.type == 'start']
		self.finish_holds = [h.id for h in self.holds.values() if h.type == 'finish']
		
	@classmethod
	def load_from_json(cls, data: dict) -> "Problem":
		"""Loads a Problem instance from structured JSON input."""
		
		# --- Compute scale factor from wall calibration data ---
		scale_data = data["wall"]["scale"]
		real_cm = scale_data.get("realDistanceCm", 0)
		pixel_dist = scale_data.get("pixelDistance", 1) or 1  # prevent div-by-zero
		scale_factor = real_cm / pixel_dist

		# --- Parse holds ---
		def parse_hold(hold: dict) -> Hold:
				pos_px = Point(hold["position"]["x"], hold["position"]["y"])
				pos_cm = Point(pos_px.x * scale_factor, pos_px.y * scale_factor)
				return Hold(
					id=hold["id"],
					type=hold["type"],
					pixel_position=pos_px,
					real_position=pos_cm
				)

		holds = {h["id"]: parse_hold(h) for h in data["holds"]}

		# --- Parse climber ---
		climber_data = data["climber"]
		climber = Climber(
				height_cm=climber_data["heightCm"],
				arm_span_cm=climber_data["armSpanCm"]
		)
		climber.set_scale_and_calculate(scale_factor)

		# --- Final assembly ---
		return cls(
				problem_name=data.get("problem_name", "Unnamed Problem"),
				climber=climber,
				holds=holds,
				scale_factor=scale_factor
		)


# =============================================================================
# Utils
# =============================================================================

def euclidian_distance(p1: Point, p2: Point) -> float:
	"""
	Calculates the straight line distance between two points.
	"""
	return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


# =============================================================================
# Kinematics
# =============================================================================
def compute_bent_joint_position(
	anchor: Point,
	target: Point,
	total_length: float,
	bulge_factor: float,
	side: Literal["left", "right"]
) -> Optional[Point]: 

	"""
    Approximates the position of an elbow/knee joint (joint between two segments)
    using an arc-based inverse kinematics (IK) approximation that always bends outward.
    
    Parameters:
    - anchor: fixed limb attachment point (e.g. shoulder/hip)
    - target: end effector (e.g. hand/foot)
    - total_length: full segment chain length (upper + lower limb), in pixels
    - bulge_factor: how far the joint "bulges" outward from the center line
    - side: determines outward direction for visual realism ('left' or 'right')
    
    Returns:
    - A Point representing the joint position (elbow/knee) in pixels
  """
	dx = target.x - anchor.x
	dy = target.y - anchor.y
	distance = math.hypot(dx, dy)

	# If the target is unreachable or trivially close, clamp it to max reach
	if distance <= 1e-6 or distance >= total_length:
		clamped_frac = min(1.0, total_length / (distance or 1.0))
		return Point(
				anchor.x + clamped_frac * dx,
				anchor.y + clamped_frac * dy
		)
	
	# --- Midpoint between anchor and target ---
	mid_x = anchor.x + dx * 0.5
	mid_y = anchor.y + dy * 0.5

	# --- Two unit-length perpendicular directions ---
	perp_a = (-dy / distance,  dx / distance)
	perp_b = ( dy / distance, -dx / distance)

	 # --- Choose the one that bends OUTWARD visually ---
	if side == "left":
			perp = perp_a if perp_a[0] < 0 else perp_b
	else:
			perp = perp_a if perp_a[0] > 0 else perp_b

	# --- Offset the joint away from the line by a bulge amount ---
	bulge = bulge_factor * distance
	joint_x = mid_x + perp[0] * bulge
	joint_y = mid_y + perp[1] * bulge

	return Point(joint_x, joint_y)

def compute_joint_angle(anchor: Point, joint: Point, target: Point) -> float:
	"""
	Calculates the angle (in degrees) at a joint formed by two segments:
	anchor → joint → target. Useful for enforcing or visualizing joint limits.

	Parameters:
	- anchor: base of first segment (e.g., shoulder)
	- joint: the vertex point (e.g., elbow)
	- target: end of second segment (e.g., hand)

	Returns:
	- Angle in degrees between the two segments.
	"""
	# Vectors from joint to anchor and joint to target
	vec1 = (anchor.x - joint.x, anchor.y - joint.y)
	vec2 = (target.x - joint.x, target.y - joint.y)

	# Dot product and magnitudes
	dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
	magnitude = math.hypot(*vec1) * math.hypot(*vec2)

	# Clamp the angle to avoid domain errors from float precision
	cosine_theta = max(-1.0, min(1.0, dot_product / (magnitude or 1.0)))
	return math.degrees(math.acos(cosine_theta))

# =============================================================================
# A* SOLVER
# =============================================================================
import heapq
import logging
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)

class AStarSolver:
	"""Solves for the optimal climbing beta using A* with reach and spatial constraints."""

	def __init__(self, problem: Problem, *, verbose: bool = False):
			self.problem = problem
			level = logging.DEBUG if verbose else logging.INFO
			logging.basicConfig(format="%(message)s", level=level, force=True)
			logger.info("--- A* solver initialized (verbose=%s) ---", verbose)

	def _get_initial_state(self) -> Optional[State]:
			"""Construct a plausible starting pose based on start hold(s)."""
			starts = self.problem.start_holds
			if len(starts) not in (1, 2):
					logger.warning("Unsupported number of start holds (%s)", len(starts))
					return None

			# --- Case: Single start hold → both hands start there ---
			if len(starts) == 1:
					start_id = starts[0]
					start_pos = self.problem.holds[start_id].real_position

					# Find footholds below the hand position (lowest two by x-distance)
					footholds = [
							h for h in self.problem.holds.values()
							if h.real_position.y > start_pos.y and h.id not in self.problem.finish_holds
					]
					if not footholds:
							logger.warning("No footholds below the single start hold – unsolvable")
							return None

					footholds.sort(key=lambda h: abs(h.real_position.x - start_pos.x))
					f1, f2 = footholds[0], footholds[1] if len(footholds) > 1 else footholds[0]
					lf, rf = (f1.id, f2.id) if f1.real_position.x <= f2.real_position.x else (f2.id, f1.id)
					return State(lh_id=start_id, rh_id=start_id, lf_id=lf, rf_id=rf)

			# --- Case: Two start holds → assign hands left/right by x position ---
			h1, h2 = (self.problem.holds[starts[0]], self.problem.holds[starts[1]])
			lh_id, rh_id = (h1.id, h2.id) if h1.real_position.x <= h2.real_position.x else (h2.id, h1.id)

			lh_y = self.problem.holds[lh_id].real_position.y
			rh_y = self.problem.holds[rh_id].real_position.y
			max_hand_y = min(lh_y, rh_y)  # Higher of the two hands (lower y = higher on wall)

			# Footholds must be below both hands
			footholds = [
					h for h in self.problem.holds.values()
					if h.real_position.y > max_hand_y and h.id not in starts and h.id not in self.problem.finish_holds
			]
			if not footholds:
					logger.warning("No footholds below the two start holds – unsolvable")
					return None

			# Closest footholds to each hand, ordered left to right
			lf = min(footholds, key=lambda h: abs(h.real_position.x - self.problem.holds[lh_id].real_position.x))
			rf = min((h for h in footholds if h.id != lf.id),
								key=lambda h: abs(h.real_position.x - self.problem.holds[rh_id].real_position.x),
								default=lf)

			if lf.real_position.x > rf.real_position.x:
					lf, rf = rf, lf

			return State(lh_id=lh_id, rh_id=rh_id, lf_id=lf.id, rf_id=rf.id)

	def _is_valid_state(self, state: State) -> bool:
			"""Validates physical plausibility: no crossovers, feet below hands."""
			holds = self.problem.holds
			lh_x = holds[state.lh_id].real_position.x
			rh_x = holds[state.rh_id].real_position.x
			lf_x = holds[state.lf_id].real_position.x
			rf_x = holds[state.rf_id].real_position.x

			# Enforce no limb crossovers (left must stay left of right)
			if not (lh_x <= rh_x and lf_x <= rf_x):
					return False

			# Enforce that feet are positioned below the hands
			max_hand_y = max(holds[state.lh_id].real_position.y, holds[state.rh_id].real_position.y)
			if holds[state.lf_id].real_position.y <= max_hand_y or holds[state.rf_id].real_position.y <= max_hand_y:
					return False

			return True

	def _is_goal_state(self, state: State) -> bool:
			"""A goal state is one where both hands are on finish holds."""
			return state.lh_id in self.problem.finish_holds and state.rh_id in self.problem.finish_holds

	def _heuristic(self, state: State) -> float:
			"""Estimated remaining cost: distance from hand midpoint to top finish hold."""
			lh = self.problem.holds[state.lh_id].real_position
			rh = self.problem.holds[state.rh_id].real_position
			mid = Point((lh.x + rh.x) / 2, (lh.y + rh.y) / 2)

			highest_finish = min(
					(self.problem.holds[f].real_position for f in self.problem.finish_holds),
					key=lambda p: p.y
			)
			return euclidian_distance(mid, highest_finish)

	def _get_successor_states(self, state: State) -> List[State]:
			"""Generate all valid next poses by moving one limb within reach."""
			successors = []
			current = {
					'lh_id': state.lh_id,
					'rh_id': state.rh_id,
					'lf_id': state.lf_id,
					'rf_id': state.rf_id,
			}

			for limb, current_hold_id in current.items():
					origin = self.problem.holds[current_hold_id].real_position
					is_arm = limb in ('lh_id', 'rh_id')
					reach = self.problem.climber.max_arm_reach_cm if is_arm else self.problem.climber.max_leg_reach_cm

					for candidate in self.problem.holds.values():
							if candidate.id == current_hold_id:
									continue
							if euclidian_distance(origin, candidate.real_position) <= reach:
									new_state = dict(current)
									new_state[limb] = candidate.id
									successors.append(State(**new_state))
			return successors

	def _calculate_move_cost(self, from_state: State, to_state: State) -> float:
			"""Cost = distance moved by whichever limb changed."""
			for limb in ['lh_id', 'rh_id', 'lf_id', 'rf_id']:
					old, new = getattr(from_state, limb), getattr(to_state, limb)
					if old != new:
							p1 = self.problem.holds[old].real_position
							p2 = self.problem.holds[new].real_position
							return euclidian_distance(p1, p2)
			return 0.0

	def _reconstruct_path(self, came_from: Dict[State, State], current: State) -> List[State]:
			"""Backtrack from goal to start to reconstruct the full solution path."""
			path = [current]
			while current in came_from:
					current = came_from[current]
					path.insert(0, current)
			return path

	def solve(self) -> Optional[List[State]]:
			"""Run the A* search and return an optimal sequence of states."""
			start = self._get_initial_state()
			if not start:
					logger.error("Failed to generate a start state – aborting search")
					return None

			open_heap = []
			entry_id = 0
			g_score = {start: 0.0}
			f_score = {start: self._heuristic(start)}
			open_set = {start}
			came_from: Dict[State, State] = {}

			heapq.heappush(open_heap, (f_score[start], entry_id, start))

			while open_heap:
					_, _, current = heapq.heappop(open_heap)
					open_set.remove(current)

					if self._is_goal_state(current):
							return self._reconstruct_path(came_from, current)

					for neighbor in self._get_successor_states(current):
							if not self._is_valid_state(neighbor):
									continue

							tentative_g = g_score[current] + self._calculate_move_cost(current, neighbor)

							if tentative_g < g_score.get(neighbor, float('inf')):
									came_from[neighbor] = current
									g_score[neighbor] = tentative_g
									f_score[neighbor] = tentative_g + self._heuristic(neighbor)

									if neighbor not in open_set:
											entry_id += 1
											heapq.heappush(open_heap, (f_score[neighbor], entry_id, neighbor))
											open_set.add(neighbor)

			logger.warning("No path found – open set exhausted")
			return None

# =============================================================================
# VISUALIZATION
# =============================================================================

def _find_best_limb_position(
    anchor: Point,
    target_pos: Point,
    max_reach: float,
    all_holds: List[Hold],
) -> Point:
	"""
	Determines the final position for a limb's end-effector (hand/foot).

	If the target hold is within reach, it's used directly. If not, it snaps
	to the nearest reachable hold (measured by distance to the original target).
	If no other holds are reachable, it clamps the limb in free space along the
	vector from the anchor to the original target.
	"""
	distance_to_target = euclidian_distance(anchor, target_pos)
	# If target is reachable, no changes needed.
	if distance_to_target <= max_reach:
			return target_pos

	# --- Target is out of reach, find the best alternative ---

	# 1. Find all other holds that are within the limb's reach from its anchor.
	reachable_alternatives = [
			h for h in all_holds
			if euclidian_distance(anchor, h.pixel_position) <= max_reach
	]

	# 2. If alternatives exist, snap to the one closest to the *original* target.
	if reachable_alternatives:
			best_alternative = min(
					reachable_alternatives,
					key=lambda h: euclidian_distance(h.pixel_position, target_pos)
			)
			return best_alternative.pixel_position

	# 3. Fallback: No reachable holds found, so clamp in free space.
	scale = max_reach / (distance_to_target or 1.0)  # Avoid division by zero
	return Point(
			anchor.x + (target_pos.x - anchor.x) * scale,
			anchor.y + (target_pos.y - anchor.y) * scale
	)

def generate_visualization_data(problem: Problem, path: List[State]) -> List[Dict[str, Any]]:
	"""
	Converts a list of climbing states into visual slides for animation or step-through display.
	Each slide contains joint positions and a description of what changed.
	"""
	slides = []

	# --- First slide: Starting pose ---
	slides.append(_create_slide_from_state(problem, path[0], "Start position."))

	# --- Subsequent slides: Describe each move ---
	for i in range(1, len(path)):
		from_state, to_state = path[i - 1], path[i]

		limb_map = {
				'lh_id': "Left Hand", 'rh_id': "Right Hand",
				'lf_id': "Left Foot", 'rf_id': "Right Foot"
		}

		# --- Collect all moves that occurred in this step ---
		move_descriptions = []
		for limb_attr, limb_name in limb_map.items():
			from_hold = getattr(from_state, limb_attr)
			to_hold = getattr(to_state, limb_attr)
			
			if from_hold != to_hold:
					move_descriptions.append(
							f"{limb_name} from hold '{from_hold}' to hold '{to_hold}'"
					)

		# --- Format the final, comprehensive description ---
		if not move_descriptions:
				description = "Unknown movement."
		else:
				# Join multiple moves with "and" for a natural sentence structure.
				description = "Move " + " and ".join(move_descriptions) + "."

		slides.append(_create_slide_from_state(problem, to_state, description, move_number=i))

	return slides

def _create_slide_from_state(
    problem: Problem,
    state: State,
    description: str,
    move_number: int = 0
) -> Dict[str, Any]:
	
	"""
	Creates a single visualization slide from a climber state.
	Calculates realistic limb joint positions using inverse kinematics (IK),
	with clamping and snapping to prevent over-extension.
	"""
	holds = problem.holds
	climber = problem.climber
	scale = problem.scale_factor

	# --- Get raw hand/foot positions from holds (in pixels) ---
	lh_pos = holds[state.lh_id].pixel_position
	rh_pos = holds[state.rh_id].pixel_position
	lf_pos = holds[state.lf_id].pixel_position
	rf_pos = holds[state.rf_id].pixel_position

	# --- Use precomputed climber segment lengths (in pixels) ---
	total_arm_px = climber.max_arm_reach_px
	total_leg_px = climber.max_leg_reach_px
	torso_height_px = (climber.height_cm * 0.25) / scale  # rough torso estimate

	# --- Compute body centerline: shoulders and hips ---
	hand_mid = Point((lh_pos.x + rh_pos.x) / 2, (lh_pos.y + rh_pos.y) / 2)
	shoulder_mid = Point(hand_mid.x, hand_mid.y + (climber.height_cm * 0.08 / scale))
	hip_mid = Point(shoulder_mid.x, shoulder_mid.y + torso_height_px)

	# --- [NEW] Calculate head position and size ---
	head_radius_px = (climber.height_cm * 0.08) / scale
	head_center = Point(shoulder_mid.x, shoulder_mid.y - head_radius_px)

	# --- Widths for shoulder and hip (based on arm span) ---
	shoulder_width_px = (climber.arm_span_cm * 0.18) / scale
	hip_width_px = shoulder_width_px * 0.9  # slightly narrower than shoulders

	# --- Joint anchor points (fixed base positions) ---
	l_shoulder = Point(shoulder_mid.x - shoulder_width_px / 2, shoulder_mid.y)
	r_shoulder = Point(shoulder_mid.x + shoulder_width_px / 2, shoulder_mid.y)
	l_hip = Point(hip_mid.x - hip_width_px / 2, hip_mid.y)
	r_hip = Point(hip_mid.x + hip_width_px / 2, hip_mid.y)

	# --- Determine final limb positions with clamping and snapping ---
	all_holds_list = list(holds.values())
	final_lh_pos = _find_best_limb_position(l_shoulder, lh_pos, total_arm_px, all_holds_list)
	final_rh_pos = _find_best_limb_position(r_shoulder, rh_pos, total_arm_px, all_holds_list)
	final_lf_pos = _find_best_limb_position(l_hip, lf_pos, total_leg_px, all_holds_list)
	final_rf_pos = _find_best_limb_position(r_hip, rf_pos, total_leg_px, all_holds_list)

	# --- Calculate elbow/knee positions using the final, validated limb positions ---
	l_elbow = compute_bent_joint_position(l_shoulder, final_lh_pos, total_arm_px, 0.3, "left")
	r_elbow = compute_bent_joint_position(r_shoulder, final_rh_pos, total_arm_px, 0.3, "right")
	l_knee = compute_bent_joint_position(l_hip, final_lf_pos, total_leg_px, 0.25, "left")
	r_knee = compute_bent_joint_position(r_hip, final_rf_pos, total_leg_px, 0.25, "right")

	# --- Convert to simple dict format for rendering/output ---
	def to_dict(p: Point) -> Dict[str, float]:
			return {"x": p.x, "y": p.y}

	return {
			"moveNumber": move_number,
			"description": description,
			"skeleton": {
				"l_shoulder": to_dict(l_shoulder),
				"r_shoulder": to_dict(r_shoulder),
				"l_hip":      to_dict(l_hip),
				"r_hip":      to_dict(r_hip),
				"l_elbow":    to_dict(l_elbow),
				"r_elbow":    to_dict(r_elbow),
				"l_knee":     to_dict(l_knee),
				"r_knee":     to_dict(r_knee),
				"l_hand":     to_dict(final_lh_pos),
				"r_hand":     to_dict(final_rh_pos),
				"l_foot":     to_dict(final_lf_pos),
				"r_foot":     to_dict(final_rf_pos),
				# --- Add head data to the skeleton payload ---
				"head_center": to_dict(head_center),
				"head_radius": head_radius_px,
			}
	}

# =============================================================================
# DRAWING & PDF GENERATION
# =============================================================================

PDF_CONFIG = {
    "json_file": "RedV5.json",
    "image_file": "assets/mockImage.png",
    "output_pdf": "RedV5_beta_slideshow.pdf",
    "font_file": "arial.ttf",
}

SKELETON_COLORS = {
    "torso": "white",
    "arm": "#87CEEB",  # Sky Blue
    "leg": "#98FB98",  # Pale Green
    "joint": "red",
	"head": (224, 224, 224, 0),   # Light grey with transparency
}
TEXT_BG_COLOR = (0, 0, 0, 150)
TEXT_COLOR = (255, 255, 255)

def _load_font(size: int = 36):
    """Try to load a custom font, fallback to default if unavailable."""
    try:
        return ImageFont.truetype(PDF_CONFIG["font_file"], size)
    except (IOError, OSError):
        print(f"⚠️ Font '{PDF_CONFIG['font_file']}' not found – using default system font")
        return ImageFont.load_default()

def draw_skeleton_on_image(image: Image.Image, slide_data: dict) -> Image.Image:
	"""
	Draws a stick-figure skeleton on top of the climbing wall image,
	using joint positions and caption info from the slide data.
	"""
	head_texture = Image.open("assets/climber_face").convert("RGBA")

	frame = image.copy()
	draw = ImageDraw.Draw(frame, "RGBA")
	skeleton = slide_data.get("skeleton")
	if not skeleton:
			return frame  # No skeleton data available

	# --- Drawing helpers ---
	def draw_line(p1_key, p2_key, color, width):
			"""Draw a limb or torso segment between two joints."""
			p1, p2 = skeleton.get(p1_key), skeleton.get(p2_key)
			if p1 and p2:
					draw.line([(p1["x"], p1["y"]), (p2["x"], p2["y"])], fill=color, width=width)

	def draw_joint(p_key, radius):
			"""Draw a filled circle at a joint."""
			p = skeleton.get(p_key)
			if p:
					draw.ellipse(
							(p["x"] - radius, p["y"] - radius, p["x"] + radius, p["y"] + radius),
							fill=SKELETON_COLORS["joint"]
					)

	# --- Draw arms ---
	draw_line("l_shoulder", "l_elbow", SKELETON_COLORS["arm"], 4)
	draw_line("l_elbow", "l_hand", SKELETON_COLORS["arm"], 4)
	draw_line("r_shoulder", "r_elbow", SKELETON_COLORS["arm"], 4)
	draw_line("r_elbow", "r_hand", SKELETON_COLORS["arm"], 4)

	# --- Draw legs ---
	draw_line("l_hip", "l_knee", SKELETON_COLORS["leg"], 4)
	draw_line("l_knee", "l_foot", SKELETON_COLORS["leg"], 4)
	draw_line("r_hip", "r_knee", SKELETON_COLORS["leg"], 4)
	draw_line("r_knee", "r_foot", SKELETON_COLORS["leg"], 4)

	# --- Draw torso using lines between shoulders and hips ---
	l_shoulder = skeleton.get("l_shoulder")
	r_shoulder = skeleton.get("r_shoulder")
	l_hip = skeleton.get("l_hip")
	r_hip = skeleton.get("r_hip")

	if l_shoulder and r_shoulder and l_hip and r_hip:
		# Connect shoulders
		draw_line("l_shoulder", "r_shoulder", SKELETON_COLORS["torso"], 4)
		# Connect hips
		draw_line("l_hip", "r_hip", SKELETON_COLORS["torso"], 4)
		# Connect left shoulder to left hip
		draw_line("l_shoulder", "l_hip", SKELETON_COLORS["torso"], 4)
		# Connect right shoulder to right hip
		draw_line("r_shoulder", "r_hip", SKELETON_COLORS["torso"], 4)


	# --- Draw the head on top of the torso ---
	head_center = skeleton.get("head_center")
	head_radius = skeleton.get("head_radius")

	if head_center and head_radius:
			# Calculate bounding box
			left = int(head_center["x"] - head_radius)
			top = int(head_center["y"] - head_radius)
			size = int(2 * head_radius)

			# Resize and adjust texture
			resized_texture = head_texture.resize((size, size)).copy()

			# Make the texture 50% transparent
			alpha = resized_texture.getchannel("A").point(lambda p: int(p * 0.7))
			resized_texture.putalpha(alpha)

			# Create an elliptical mask
			mask = Image.new("L", (size, size), 0)
			mask_draw = ImageDraw.Draw(mask)
			mask_draw.ellipse((0, 0, size, size), fill=255)

			# Apply elliptical clip mask to alpha channel
			texture_with_mask = Image.new("RGBA", (size, size), (0, 0, 0, 0))
			texture_with_mask.paste(resized_texture, (0, 0), mask)

			# Paste onto the frame using alpha_composite
			# This requires creating a subregion the size of the texture
			region = frame.crop((left, top, left + size, top + size))
			blended = Image.alpha_composite(region, texture_with_mask)
			frame.paste(blended, (left, top))

	# --- Draw joint circles last so they appear on top of limbs ---
	for key, value in skeleton.items():
			# Only draw points (which are dicts), and exclude the head's center point
			if isinstance(value, dict) and key != "head_center":
					draw_joint(key, radius=7)

	# --- Add description text at bottom ---
	font = _load_font(36)
	description = slide_data.get("description", "")
	bbox = draw.textbbox((0, 0), description, font=font)
	text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
	text_x = (frame.width - text_w) / 2
	text_y = frame.height - text_h - 30

	# Semi-transparent black background behind text
	draw.rectangle(
			(text_x - 15, text_y - 10, text_x + text_w + 15, text_y + text_h + 10),
			fill=TEXT_BG_COLOR
	)
	draw.text((text_x, text_y), description, font=font, fill=TEXT_COLOR)

	return frame

def generate_pdf_slideshow():
    """
    Generate a beta slideshow PDF using A* pathfinding from fixed input files.
    Assumes RedV5.json and mockImage.png exist.
    """
    print("\n--- Generating PDF Slideshow ---")

    # Load problem JSON
    with open(PDF_CONFIG["json_file"], "r") as fh:
        problem_data = json.load(fh)

    # Load climbing wall background image
    base_image = Image.open(PDF_CONFIG["image_file"]).convert("RGBA")

    # Solve for optimal climbing beta path
    problem = Problem.load_from_json(problem_data)
    solver = AStarSolver(problem)
    print("Solving for optimal beta route...")
    path = solver.solve()

    if not path:
        print("❌ No solution found.")
        return

    print(f"✅ Found solution with {len(path)} steps. Rendering frames...")

    # Generate skeleton overlay data for each step
    slides = generate_visualization_data(problem, path)
    frames = [
        draw_skeleton_on_image(base_image, slide).convert("RGB")
        for slide in slides
    ]

    # Save all frames to a PDF
    frames[0].save(
        PDF_CONFIG["output_pdf"],
        save_all=True,
        append_images=frames[1:],
        title=f"Beta Slideshow for {problem.problem_name}",
    )

    print(f"✨ PDF saved to: {PDF_CONFIG['output_pdf']}")
    print("--- Done ---")


if __name__ == "__main__":
    generate_pdf_slideshow()