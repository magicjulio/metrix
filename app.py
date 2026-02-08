from flask import Flask, render_template, request, jsonify, url_for
import numpy as np
import os
import subprocess
import tempfile
import uuid
import textwrap
import sys
import shutil
from calculations import analyze_matrix  # Import your new module

app = Flask(__name__)

MIN_DIM = 1
MAX_DIM = 10
DEFAULT_ROWS = 3
DEFAULT_COLS = 3

def clamp_dim(value: int) -> int:
    """Clamp a dimension to the allowed bounds."""
    return max(MIN_DIM, min(value, MAX_DIM))


def _format_num(val: float) -> str:
    if np.isclose(val, np.round(val), atol=1e-8):
        return str(int(np.round(val)))
    return f"{val:.2f}".rstrip("0").rstrip(".")


def _build_manim_scene_code(matrix: np.ndarray) -> str:
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    a_s, b_s, c_s, d_s = map(_format_num, [a, b, c, d])
    print(a_s, b_s, c_s, d_s)
    tex_matrix = f"\\begin{{pmatrix}} {a_s} & {b_s} \\\\ {c_s} & {d_s} \\end{{pmatrix}}"
    return textwrap.dedent(
        f"""
        from manim import *


        class MatrixTransform2D(LinearTransformationScene):
            def __init__(self):
                super().__init__(
                    show_coordinates=True,
                    leave_ghost_vectors=True,
                    show_basis_vectors=True,
                )

            def construct(self):
                matrix = [[{a_s}, {b_s}], [{c_s}, {d_s}]]
                matrix_tex = MathTex(r"A = {tex_matrix}")
                matrix_tex.to_corner(UL).add_background_rectangle()
                self.add(matrix_tex)
                self.wait(0.5)
                self.apply_matrix(matrix)
                self.wait(1)
        """
    )


def generate_animation(matrix: np.ndarray) -> str:
    static_root = os.path.join(app.root_path, "static")
    output_dir = os.path.join(static_root, "videos")
    os.makedirs(output_dir, exist_ok=True)

    # Use a unique temporary directory to sandbox Manim execution
    with tempfile.TemporaryDirectory() as tmp_dir:
        scene_path = os.path.join(tmp_dir, "scene.py")
        
        with open(scene_path, "w", encoding="utf-8") as f:
            f.write(_build_manim_scene_code(matrix))

        output_name = "matrix_anim.mp4"
        
        cmd = [
            sys.executable,
            "-m",
            "manim",
            "-ql",
            "--disable_caching",
            "scene.py",
            "MatrixTransform2D",
            "-o",
            output_name,
            "--media_dir", ".",  # Keep all generated media inside tmp_dir
        ]
        
        # Run with cwd=tmp_dir to capture any default output files
        subprocess.run(cmd, check=True, cwd=tmp_dir, capture_output=True, text=True)

        found_path = None
        for root, _, files in os.walk(tmp_dir):
            if output_name in files:
                found_path = os.path.join(root, output_name)
                break

        if not found_path:
            raise FileNotFoundError("Manim output video not found.")

        final_path = os.path.join(output_dir, output_name)
        try:
            os.remove(final_path)
        except OSError:
            pass
        shutil.move(found_path, final_path)

        rel_path = os.path.relpath(final_path, static_root)
        cache_bust = uuid.uuid4().hex
        return url_for("static", filename=rel_path) + f"?v={cache_bust}"


def _cleanup_old_videos(media_dir: str) -> None:
    for root, _, files in os.walk(media_dir):
        for file in files:
            if file.endswith(".mp4"):
                path = os.path.join(root, file)
                try:
                    os.remove(path)
                except OSError:
                    pass

    # Remove empty directories (bottom-up)
    for root, dirs, files in os.walk(media_dir, topdown=False):
        if not dirs and not files:
            try:
                os.rmdir(root)
            except OSError:
                pass

@app.route('/', methods=['GET', 'POST'])
def home():
    """Render the landing page with an n×m matrix input grid (3×3 uses Eigenraum(λ=1))."""
    submitted_array = None
    results = None  # Initialize results context
    rows = DEFAULT_ROWS
    cols = DEFAULT_COLS

    if request.method == "POST":
        try:
            rows = int(request.form.get("rows", rows))
            cols = int(request.form.get("cols", cols))
        except ValueError:
            rows, cols = DEFAULT_ROWS, DEFAULT_COLS

        rows = clamp_dim(rows)
        cols = clamp_dim(cols)

        raw_values = []
        
        for row in range(1, rows + 1):
            row_entries = []
            for col in range(1, cols + 1):
                raw = request.form.get(f"a{row}{col}", "").strip()
                if not raw:
                    row_entries.append(0.0) # Default to 0 instead of NaN for cleaner math
                else:
                    try:
                        row_entries.append(float(raw))
                    except ValueError:
                        row_entries.append(0.0)
                        
            raw_values.append(row_entries)
        
        submitted_array = np.array(raw_values, dtype=float)
        
        # Calculate all metrics in one go using the imported function
        results = analyze_matrix(submitted_array)
        
    return render_template(
        "index.html",
        submitted_array=submitted_array,
        results=results,  # Pass the entire dictionary
        rows=rows,
        cols=cols,
        min_dim=MIN_DIM,
        max_dim=MAX_DIM
    )


@app.post("/animate")
def animate():
    data = request.get_json(silent=True) or {}
    try:
        rows = int(data.get("rows", 0))
        cols = int(data.get("cols", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid matrix size."}), 400

    if rows != 2 or cols != 2:
        return jsonify({"error": "Animation is available for 2×2 matrices only."}), 400

    values = data.get("values")
    if not values or len(values) != 2 or any(len(r) != 2 for r in values):
        return jsonify({"error": "Invalid matrix values."}), 400

    try:
        matrix = np.array(values, dtype=float)
    except (TypeError, ValueError):
        return jsonify({"error": "Matrix values must be numeric."}), 400

    try:
        video_url = generate_animation(matrix)
    except subprocess.CalledProcessError as exc:
        app.logger.exception("Manim failed: %s", exc.stderr or exc.stdout)
        return jsonify({
            "error": "Failed to generate animation.",
            "details": (exc.stderr or exc.stdout or "No error output from Manim.")
        }), 500
    except Exception:
        app.logger.exception("Unexpected animation error")
        return jsonify({"error": "Failed to generate animation."}), 500

    return jsonify({"video_url": video_url})

if __name__ == "__main__":
    app.run(host="0.0.0.0")