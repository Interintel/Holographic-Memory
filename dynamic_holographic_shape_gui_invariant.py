import json
import os
import hashlib
import tkinter as tk
from tkinter import messagebox
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np


# ============================================================
# Dynamic Holographic Shape Memory with Invariance
# ------------------------------------------------------------
# Timothy Busbice, (c) 2026
# tbusbice@connectomicagi.com
#
# Draw a shape with your cursor.
# Label it.
# Store the shape + label in a dynamic holographic data file.
# Later, draw another shape and let the system recall the label.
#
# Invariance upgrades:
#   - Translation invariance: shape position on canvas does not matter.
#   - Scale invariance: large and small versions compare similarly.
#   - Rotation invariance: tilted shapes still match.
#   - Stroke-start invariance: where you began drawing does not matter.
#   - Direction invariance: clockwise/counter-clockwise does not matter.
#   - Optional mirror invariance: flipped shapes can still match.
#
# Requires:
#   pip install numpy
#
# Tkinter usually comes with Python.
# ============================================================


MEMORY_FILE = "dynamic_holographic_shape_memory_invariant.npz"


@dataclass
class MemoryConfig:
    hrr_dim: int = 512
    radial_bins: int = 128
    point_resample_count: int = 256

    # Invariance controls
    rotation_invariant: bool = True
    mirror_invariant: bool = True
    rotation_steps: int = 32

    # Lower threshold = more willing to guess.
    # Higher threshold = stricter match.
    similarity_threshold: float = 0.22

    random_seed: int = 42


class HRR:
    """
    Holographic Reduced Representation helper.

    Circular convolution binds vectors.
    Cosine similarity compares vectors.
    """

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            return v
        return v / norm

    @staticmethod
    def circular_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real.astype(np.float32)

    @staticmethod
    def circular_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.fft.ifft(np.conj(np.fft.fft(a)) * np.fft.fft(b)).real.astype(np.float32)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-9:
            return 0.0
        return float(np.dot(a, b) / denom)


class ShapeEncoder:
    """
    Converts cursor-drawn points into a fixed-size shape signature,
    then projects that signature into an HRR vector.

    The main feature is a radial signature:
    - Find the center of the drawing.
    - Convert points to angles around the center.
    - Store the farthest radius observed in each angle bin.
    - Normalize so size and position matter less.

    Rotation invariance is added by comparing cyclic shifts of the
    radial signature. A rotated shape is basically the same radial
    pattern shifted around the circular angle bins.
    """

    def __init__(self, config: MemoryConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.random_seed)

        self.projection = self.rng.normal(
            0,
            1,
            size=(config.radial_bins, config.hrr_dim)
        ).astype(np.float32)

        self.projection /= (
            np.linalg.norm(self.projection, axis=1, keepdims=True) + 1e-9
        )

    def resample_points(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Resamples the cursor path to a fixed number of points.
        This helps make quick and slow drawings comparable.
        """

        if len(points) < 2:
            return np.zeros((self.cfg.point_resample_count, 2), dtype=np.float32)

        pts = np.array(points, dtype=np.float32)

        distances = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
        cumulative = np.insert(np.cumsum(distances), 0, 0.0)

        total_length = cumulative[-1]

        if total_length < 1e-6:
            return np.repeat(pts[:1], self.cfg.point_resample_count, axis=0)

        target = np.linspace(0, total_length, self.cfg.point_resample_count)
        x = np.interp(target, cumulative, pts[:, 0])
        y = np.interp(target, cumulative, pts[:, 1])

        return np.stack([x, y], axis=1).astype(np.float32)

    def normalize_points(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Makes the drawing translation and scale invariant.

        Translation invariance:
            subtract center

        Scale invariance:
            divide by maximum radius
        """

        pts = self.resample_points(points)

        center = np.mean(pts, axis=0)
        shifted = pts - center

        radii = np.sqrt(np.sum(shifted ** 2, axis=1))
        max_radius = float(np.max(radii))

        if max_radius < 1e-6:
            return shifted

        return shifted / max_radius

    def extract_signature(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Converts a drawing into a normalized radial signature.

        This signature is already:
        - translation invariant
        - scale invariant
        - mostly stroke-speed invariant
        - stroke-start invariant
        """

        shifted = self.normalize_points(points)

        radii = np.sqrt(np.sum(shifted ** 2, axis=1))

        if float(np.max(radii)) < 1e-6:
            return np.zeros(self.cfg.radial_bins, dtype=np.float32)

        angles = np.arctan2(shifted[:, 1], shifted[:, 0])
        angles = (angles + 2 * np.pi) % (2 * np.pi)

        bins = np.floor(
            angles / (2 * np.pi) * self.cfg.radial_bins
        ).astype(int)

        bins = np.clip(bins, 0, self.cfg.radial_bins - 1)

        signature = np.zeros(self.cfg.radial_bins, dtype=np.float32)

        for bin_index, radius in zip(bins, radii):
            signature[bin_index] = max(signature[bin_index], radius)

        # Fill empty bins and soften jagged hand motion.
        for _ in range(4):
            left = np.roll(signature, 1)
            right = np.roll(signature, -1)
            signature = np.maximum(signature, 0.50 * (left + right))

        # Normalize the signature distribution.
        signature = signature - np.mean(signature)
        std = np.std(signature)

        if std > 1e-6:
            signature = signature / std

        return signature.astype(np.float32)

    def signature_to_hrr(self, signature: np.ndarray) -> np.ndarray:
        """
        Projects a shape signature into holographic vector space.
        """

        projected = signature @ self.projection
        projected = projected.astype(np.float32)

        return HRR.normalize(projected)

    def make_invariant_signatures(self, signature: np.ndarray) -> List[np.ndarray]:
        """
        Creates rotated and optionally mirrored versions of a shape signature.

        Rotation invariance:
            use cyclic shifts of the radial signature.

        Mirror invariance:
            reverse the circular signature, then also rotate it.
        """

        signatures = []

        if self.cfg.rotation_invariant:
            step_count = max(1, int(self.cfg.rotation_steps))
            shifts = np.linspace(
                0,
                self.cfg.radial_bins,
                step_count,
                endpoint=False
            ).astype(int)
        else:
            shifts = np.array([0], dtype=int)

        for shift in shifts:
            signatures.append(np.roll(signature, shift).astype(np.float32))

        if self.cfg.mirror_invariant:
            mirrored = signature[::-1].astype(np.float32)

            for shift in shifts:
                signatures.append(np.roll(mirrored, shift).astype(np.float32))

        return signatures

    def encode_shape(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Encodes the original non-rotated signature.
        """

        signature = self.extract_signature(points)
        return self.signature_to_hrr(signature)

    def encode_shape_variants(self, points: List[Tuple[float, float]]) -> List[np.ndarray]:
        """
        Encodes all invariant variants of the drawing.
        These variants are used during recall.
        """

        signature = self.extract_signature(points)
        signatures = self.make_invariant_signatures(signature)

        return [self.signature_to_hrr(sig) for sig in signatures]


class DynamicHolographicShapeMemory:
    """
    Stores labeled shape examples in a dynamic holographic memory file.

    Each saved example contains:
    - label text
    - base shape HRR vector
    - label HRR vector
    - bound HRR vector = shape_vector circular_convolution label_vector

    Recognition:
    - Draw a query shape.
    - Encode rotated/mirrored variants into HRR vectors.
    - Compare against stored shape vectors.
    - Return the label of the closest match.
    """

    def __init__(self, config: MemoryConfig, memory_file: str = MEMORY_FILE):
        self.cfg = config
        self.memory_file = memory_file
        self.encoder = ShapeEncoder(config)

        self.labels: List[str] = []
        self.shape_vectors: List[np.ndarray] = []
        self.label_vectors: Dict[str, np.ndarray] = {}
        self.bound_vectors: List[np.ndarray] = []

        self.dynamic_hologram = np.zeros(config.hrr_dim, dtype=np.float32)

        self.load()

    def deterministic_seed_from_text(self, text: str) -> int:
        """
        Uses SHA-256 instead of Python's hash() so label vectors are stable
        across different program runs.
        """

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little", signed=False)

    def create_label_vector(self, label: str) -> np.ndarray:
        key = label.strip().lower()

        if key in self.label_vectors:
            return self.label_vectors[key]

        seed = self.deterministic_seed_from_text(key)
        rng = np.random.default_rng(seed)

        vector = rng.normal(0, 1, size=self.cfg.hrr_dim).astype(np.float32)
        vector = HRR.normalize(vector)

        self.label_vectors[key] = vector

        return vector

    def add_example(self, points: List[Tuple[float, float]], label: str):
        label = label.strip()

        if not label:
            raise ValueError("Label cannot be empty.")

        if len(points) < 10:
            raise ValueError("Draw a larger shape before saving.")

        shape_vector = self.encoder.encode_shape(points)
        label_vector = self.create_label_vector(label)

        bound_vector = HRR.circular_convolution(shape_vector, label_vector)
        bound_vector = HRR.normalize(bound_vector)

        self.labels.append(label)
        self.shape_vectors.append(shape_vector)
        self.bound_vectors.append(bound_vector)

        # Dynamic holographic data store accumulates each bound shape-label memory.
        self.dynamic_hologram += bound_vector
        self.dynamic_hologram = HRR.normalize(self.dynamic_hologram)

        self.save()

    def recall(self, points: List[Tuple[float, float]]) -> Dict[str, Any]:
        if len(self.shape_vectors) == 0:
            return {
                "label": None,
                "similarity": 0.0,
                "message": "No shapes have been saved yet."
            }

        if len(points) < 10:
            return {
                "label": None,
                "similarity": 0.0,
                "message": "Draw a larger shape before recall."
            }

        query_variants = self.encoder.encode_shape_variants(points)

        best_index = -1
        best_similarity = -1.0

        # Compare every invariant query variant against each stored example.
        for i, stored_vector in enumerate(self.shape_vectors):
            for query_vector in query_variants:
                similarity = HRR.cosine(query_vector, stored_vector)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_index = i

        best_label = self.labels[best_index]

        if best_similarity < self.cfg.similarity_threshold:
            return {
                "label": best_label,
                "similarity": best_similarity,
                "message": (
                    f"Closest match is '{best_label}', but confidence is low. "
                    "Save more examples of each shape."
                )
            }

        return {
            "label": best_label,
            "similarity": best_similarity,
            "message": f"Recalled label: {best_label}"
        }

    def save(self):
        label_keys = list(self.label_vectors.keys())
        label_matrix = np.array(
            [self.label_vectors[k] for k in label_keys],
            dtype=np.float32
        )

        shape_matrix = (
            np.array(self.shape_vectors, dtype=np.float32)
            if self.shape_vectors
            else np.zeros((0, self.cfg.hrr_dim), dtype=np.float32)
        )

        bound_matrix = (
            np.array(self.bound_vectors, dtype=np.float32)
            if self.bound_vectors
            else np.zeros((0, self.cfg.hrr_dim), dtype=np.float32)
        )

        np.savez_compressed(
            self.memory_file,
            labels=np.array(self.labels, dtype=object),
            shape_vectors=shape_matrix,
            label_keys=np.array(label_keys, dtype=object),
            label_matrix=label_matrix,
            bound_vectors=bound_matrix,
            dynamic_hologram=self.dynamic_hologram,
            config=json.dumps(self.cfg.__dict__)
        )

    def load(self):
        if not os.path.exists(self.memory_file):
            return

        data = np.load(self.memory_file, allow_pickle=True)

        self.labels = list(data["labels"])

        shape_matrix = data["shape_vectors"]
        self.shape_vectors = [
            shape_matrix[i].astype(np.float32)
            for i in range(len(shape_matrix))
        ]

        if "bound_vectors" in data:
            bound_matrix = data["bound_vectors"]
            self.bound_vectors = [
                bound_matrix[i].astype(np.float32)
                for i in range(len(bound_matrix))
            ]

        if "dynamic_hologram" in data:
            self.dynamic_hologram = data["dynamic_hologram"].astype(np.float32)

        self.label_vectors = {}

        if "label_keys" in data and "label_matrix" in data:
            label_keys = list(data["label_keys"])
            label_matrix = data["label_matrix"]

            for key, vector in zip(label_keys, label_matrix):
                self.label_vectors[str(key)] = vector.astype(np.float32)

    def count(self) -> int:
        return len(self.labels)

    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}

        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1

        return counts

    def invariance_summary(self) -> str:
        active = [
            "translation",
            "scale",
            "stroke-start",
            "drawing-direction"
        ]

        if self.cfg.rotation_invariant:
            active.append("rotation")

        if self.cfg.mirror_invariant:
            active.append("mirror")

        return ", ".join(active)


class ShapeDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Holographic Shape Memory - Invariant")

        self.config = MemoryConfig()
        self.memory = DynamicHolographicShapeMemory(self.config)

        self.points: List[Tuple[float, float]] = []

        self.canvas_width = 760
        self.canvas_height = 520

        self.build_ui()
        self.update_status()

    def build_ui(self):
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(
            main,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white",
            cursor="crosshair"
        )

        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

        controls = tk.Frame(main)
        controls.pack(fill=tk.X, pady=8)

        tk.Label(controls, text="Shape label:").pack(side=tk.LEFT)

        self.label_entry = tk.Entry(controls, width=24)
        self.label_entry.pack(side=tk.LEFT, padx=6)

        tk.Button(
            controls,
            text="Save Shape + Label",
            command=self.save_shape
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            controls,
            text="Recall Label From Drawing",
            command=self.recall_shape
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            controls,
            text="Clear Drawing",
            command=self.clear_canvas
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            controls,
            text="Memory Stats",
            command=self.show_stats
        ).pack(side=tk.LEFT, padx=4)

        self.result_label = tk.Label(
            main,
            text="Draw a shape, type a label, then save it.",
            anchor="w",
            justify=tk.LEFT
        )
        self.result_label.pack(fill=tk.X, pady=4)

        self.status_label = tk.Label(
            main,
            text="",
            anchor="w",
            justify=tk.LEFT
        )
        self.status_label.pack(fill=tk.X)

        help_text = (
            "Invariance active: position, size, rotation, stroke-start, "
            "drawing direction, and mirror matching."
        )

        self.help_label = tk.Label(
            main,
            text=help_text,
            anchor="w",
            justify=tk.LEFT
        )
        self.help_label.pack(fill=tk.X, pady=2)

    def start_draw(self, event):
        self.points = [(event.x, event.y)]
        self.canvas.delete("result_text")
        self.result_label.config(text="Drawing...")

    def draw(self, event):
        if not self.points:
            self.points.append((event.x, event.y))
            return

        last_x, last_y = self.points[-1]
        self.points.append((event.x, event.y))

        self.canvas.create_line(
            last_x,
            last_y,
            event.x,
            event.y,
            width=3,
            fill="black",
            capstyle=tk.ROUND,
            smooth=True
        )

    def end_draw(self, event):
        if len(self.points) > 2:
            self.points.append((event.x, event.y))
            self.result_label.config(
                text=f"Drawing captured with {len(self.points)} cursor points."
            )

    def save_shape(self):
        label = self.label_entry.get().strip()

        if not label:
            messagebox.showwarning(
                "Missing Label",
                "Type a label first, like square, circle, triangle, star, etc."
            )
            return

        try:
            self.memory.add_example(self.points, label)

            self.result_label.config(
                text=f"Saved invariant shape as '{label}' into {self.memory.memory_file}"
            )

            self.draw_center_text(f"Saved: {label}")
            self.update_status()

        except Exception as exc:
            messagebox.showerror("Could Not Save", str(exc))

    def recall_shape(self):
        result = self.memory.recall(self.points)

        label = result["label"]
        similarity = result["similarity"]

        if label is None:
            self.result_label.config(text=result["message"])
            messagebox.showinfo("Recall Result", result["message"])
            return

        self.result_label.config(
            text=f"{result['message']} | invariant similarity: {similarity:.3f}"
        )

        self.draw_center_text(f"{label} ({similarity:.2f})")

    def draw_center_text(self, text):
        self.canvas.delete("result_text")
        self.canvas.create_text(
            self.canvas_width // 2,
            30,
            text=text,
            fill="blue",
            font=("Arial", 18, "bold"),
            tags="result_text"
        )

    def clear_canvas(self):
        self.canvas.delete("all")
        self.points = []
        self.result_label.config(text="Canvas cleared. Draw another shape.")

    def update_status(self):
        self.status_label.config(
            text=(
                f"Memory file: {self.memory.memory_file} | "
                f"Saved examples: {self.memory.count()} | "
                f"HRR dimension: {self.config.hrr_dim} | "
                f"Rotation variants: {self.config.rotation_steps}"
            )
        )

    def show_stats(self):
        counts = self.memory.label_counts()

        if not counts:
            messagebox.showinfo(
                "Memory Stats",
                "No shapes have been saved yet."
            )
            return

        lines = [
            "Saved shape labels:",
            ""
        ]

        for label, count in sorted(counts.items()):
            lines.append(f"{label}: {count}")

        lines.append("")
        lines.append(f"Total examples: {self.memory.count()}")
        lines.append(f"Memory file: {self.memory.memory_file}")
        lines.append("")
        lines.append("Invariance capabilities:")
        lines.append(self.memory.invariance_summary())

        messagebox.showinfo("Memory Stats", "\n".join(lines))


def main():
    root = tk.Tk()
    ShapeDrawingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
