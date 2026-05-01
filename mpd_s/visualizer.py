from typing import List, Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import EllipseCollection, LineCollection, PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import BoxStyle, Circle, FancyBboxPatch

from mpd_s.universe.environments import EnvBase
from mpd_s.universe.primitives import PrimitiveShapeField
from mpd_s.universe.robot import RobotBase


class Visualizer:
    COLORS = {
        "trajectory_collision": "red",
        "trajectory_free": "lightgreen",
        "robot_collision": "red",
        "robot_free": "green",
        "robot_collision_moving": "black",
        "robot_free_moving": "darkgreen",
        "trajectory_best": "blue",
        "robot_best": "blue",
        "start": "orange",
        "goal": "magenta",
        "fixed_obstacle": "grey",
        "extra_obstacle": "red",
    }
    
    ZORDERS = {
        "environment": 0,
        "trajectory": 1,
        "robot": 2,
        "trajectory_best": 3,
        "robot_best": 4,
        "start_goal": 5,
    }

    SIZES = {
        "robot_point_radius": 0.000,
        "start_goal_radius": 0.02,
        "trajectory_line_width": 1.0,
        "trajectory_best_line_width": 1.5,
        "robot_line_width": 0.5,
        "robot_line_width_moving": 1.0,
    }

    def __init__(
        self, env: EnvBase, robot: RobotBase, use_extra_objects: bool = True
    ) -> None:
        self.env = env
        self.robot = robot
        self.use_extra_objects = use_extra_objects

    def _render_primitive_field(
        self, ax: plt.Axes, field: PrimitiveShapeField, color: str = "gray"
    ) -> None:
        if field.name == "MultiSphere":
            centers_np = field.centers.cpu().numpy()
            radii_np = field.radii.cpu().numpy()
            patches = []
            for center_np, radius_np in zip(centers_np, radii_np):
                circle = Circle(center_np, radius_np)
                patches.append(circle)

            p = PatchCollection(patches, color=color, alpha=1.0, zorder=self.ZORDERS["environment"])
            ax.add_collection(p)

        elif field.name == "MultiBox":
            centers_np = field.centers.cpu().numpy()
            half_sizes_np = field.half_sizes.cpu().numpy()
            radii_np = field.radii.cpu().numpy()

            patches = []
            for center_np, half_size_np, radius in zip(
                centers_np, half_sizes_np, radii_np
            ):
                corner = center_np - half_size_np

                box = FancyBboxPatch(
                    corner,
                    2 * half_size_np[0],
                    2 * half_size_np[1],
                    boxstyle=BoxStyle.Round(pad=0.0, rounding_size=radius),
                )
                patches.append(box)

            p = PatchCollection(patches, color=color, zorder=self.ZORDERS["environment"])
            ax.add_collection(p)

    def _render_environment(self, ax: plt.Axes, use_extra_objects: bool = True) -> None:
        for field in self.env.obj_field_fixed.fields:
            self._render_primitive_field(ax, field, color=self.COLORS["fixed_obstacle"])

        if use_extra_objects:
            for field in self.env.obj_field_extra.fields:
                self._render_primitive_field(
                    ax, field, color=self.COLORS["extra_obstacle"]
                )
        limits = self.env.limits_np
        ax.set_xlim(limits[0][0], limits[1][0])
        ax.set_ylim(limits[0][1], limits[1][1])

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    def _compute_robot_colors(
        self,
        collision_mask: torch.Tensor,
        best_traj_idx: int = None,
        type: str = "base",
    ) -> List[List[str]]:
        colors = [
            [
                self.COLORS["robot_best"]
                if i == best_traj_idx
                else (
                    self.COLORS[
                        "robot_collision_moving" if type == "moving" else "robot_collision"
                    ]
                )
                if collision
                else (self.COLORS["robot_free_moving" if type == "moving" else "robot_free"])
                for collision in row
            ]
            for i, row in enumerate(collision_mask)
        ]
        return colors

    def _compute_trajectory_colors(
        self, trajectories_collision_mask: torch.Tensor, best_traj_idx: int = None
    ) -> List[str]:
        colors = [
            self.COLORS["trajectory_best"]
            if i == best_traj_idx
            else self.COLORS["trajectory_collision"]
            if coll
            else self.COLORS["trajectory_free"]
            for i, coll in enumerate(trajectories_collision_mask)
        ]
        return colors

    def _render_robot_pos(
        self,
        ax: plt.Axes,
        trajectories: torch.Tensor,
        colors: List[List[str]],
        type: str,
        zorder: int,
    ) -> None:
        n_trajectories, n_support_points, n_dim = trajectories.shape
        trajectories_np = trajectories.cpu().numpy()

        collision_points = self.robot.get_collision_points(trajectories)
        num_spheres = collision_points.shape[2]
        collision_points_np = collision_points.cpu().numpy()

        patches_to_draw = []
        patches_best_to_draw = []
        colors_to_draw = []
        colors_best_to_draw = []

        radius = self.SIZES["start_goal_radius"] if type == "startgoal" else self.SIZES["robot_point_radius"]

        for n in range(n_trajectories):
            for m in range(0, n_support_points):
                color = colors[n][m]
    

                for k in range(num_spheres):
                    p = collision_points_np[n, m, k]
                    if color == self.COLORS["robot_best"]:
                        patches_best_to_draw.append(Circle(p, radius))
                        colors_best_to_draw.append(color)
                    else:
                        patches_to_draw.append(Circle(p, radius))
                        colors_to_draw.append(color)

                if self.robot.name == "L2D":
                    point = trajectories_np[n, m]
                    right, base, top = point[:2], point[2:4], point[4:]

                    ax.plot(
                        [base[0], right[0]],
                        [base[1], right[1]],
                        color=color,
                        linewidth=self.SIZES["robot_line_width"],
                        zorder=zorder,
                    )
                    ax.plot(
                        [base[0], top[0]],
                        [base[1], top[1]],
                        color=color,
                        linewidth=self.SIZES["robot_line_width"],
                        zorder=zorder,
                    )

        if patches_to_draw:
            p = PatchCollection(
                patches_to_draw, facecolors=colors_to_draw, zorder=zorder
            )
            ax.add_collection(p)
        if patches_best_to_draw:
            p = PatchCollection(
                patches_best_to_draw, facecolors=colors_best_to_draw, zorder=self.ZORDERS["robot_best"]
            )
            ax.add_collection(p)

    def _render_start_goal_pos(
        self, ax: plt.Axes, start_pos: torch.Tensor, goal_pos: torch.Tensor
    ) -> None:
        start_pos_t = start_pos.reshape(1, 1, -1)
        goal_pos_t = goal_pos.reshape(1, 1, -1)

        self._render_robot_pos(
            ax, start_pos_t, colors=[[self.COLORS["start"]]], type="startgoal", zorder=self.ZORDERS["start_goal"]
        )

        self._render_robot_pos(
            ax, goal_pos_t, colors=[[self.COLORS["goal"]]], type="startgoal", zorder=self.ZORDERS["start_goal"]
        )

    def _render_trajectories(
        self,
        ax: plt.Axes,
        trajectories: torch.Tensor,
        colors: List[str],
    ) -> None:
        trajectories_np = (
            trajectories.reshape(trajectories.shape[:-1] + (-1, 2))
            .permute(2, 0, 1, 3)
            .cpu()
            .numpy()
        )
        colors_extended = [
            colors[j]
            for i in range(trajectories_np.shape[0])
            for j in range(trajectories_np.shape[1])
            if colors[j] != self.COLORS["trajectory_best"]
        ]

        segments = [
            trajectories_np[i][j]
            for i in range(trajectories_np.shape[0])
            for j in range(trajectories_np.shape[1])
            if colors[j] != self.COLORS["trajectory_best"]
        ]
        
        colors_best_extended = [
            colors[j]
            for i in range(trajectories_np.shape[0])
            for j in range(trajectories_np.shape[1])
            if colors[j] == self.COLORS["trajectory_best"]
        ]
        
        segments_best = [
            trajectories_np[i][j]
            for i in range(trajectories_np.shape[0])
            for j in range(trajectories_np.shape[1])
            if colors[j] == self.COLORS["trajectory_best"]
        ]
        line_collection = LineCollection(
            segments,
            colors=colors_extended,
            linewidths=self.SIZES["trajectory_line_width"],
            zorder=self.ZORDERS["trajectory"],
            alpha=0.7,
        )
        line_collection_best = LineCollection(
            segments_best,
            colors=colors_best_extended,
            linewidths=self.SIZES["trajectory_best_line_width"],
            zorder=self.ZORDERS["trajectory_best"],
            alpha=1.0,
        )
        ax.add_collection(line_collection)
        ax.add_collection(line_collection_best)

    def render_scene(
        self,
        trajectories: torch.Tensor,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        best_traj_idx: Optional[int] = None,
        start_pos: Optional[torch.Tensor] = None,
        goal_pos: Optional[torch.Tensor] = None,
        points_collision_mask: Optional[torch.Tensor] = None,
        draw_indices: Optional[List[int]] = None,
        draw_spacing: int = 1,
        save_path: Optional[str] = "trajectories_figure.png",
    ) -> Tuple[plt.Figure, plt.Axes]:
        draw_indices = (
            draw_indices
            if draw_indices is not None
            else list(range(trajectories.shape[0]))
        )
        if best_traj_idx is not None and best_traj_idx not in draw_indices:
            best_traj_idx = None

        if best_traj_idx is not None:
            best_traj_idx = draw_indices.index(best_traj_idx)

        trajectories = self.robot.get_position(trajectories[draw_indices])
        n_trajectories, n_support_points, n_dim = trajectories.shape

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        if points_collision_mask is None:
            _, _, points_collision_mask = (
                self.robot.get_trajectories_collision_and_free(
                    env=self.env,
                    trajectories=trajectories,
                    on_extra=self.use_extra_objects,
                )
            )
            
        trajectories_collision_mask = points_collision_mask.any(dim=-1)

        traj_colors = self._compute_trajectory_colors(
            trajectories_collision_mask, best_traj_idx
        )

        points_collision_mask = points_collision_mask[:, ::11]
        assert points_collision_mask.shape == (n_trajectories, n_support_points)

        self._render_trajectories(ax, trajectories, traj_colors)

        draw_indices_col = torch.arange(n_support_points)[draw_spacing // 2::draw_spacing]
        trajectories = trajectories[:, draw_indices_col]
        points_collision_mask = points_collision_mask[:, draw_indices_col]

        robot_pos_colors = self._compute_robot_colors(
            points_collision_mask, best_traj_idx
        )

        self._render_environment(ax, use_extra_objects=self.use_extra_objects)
        self._render_robot_pos(
            ax,
            trajectories,
            robot_pos_colors,
            type="base",
            zorder=self.ZORDERS["robot"],
        )

        if start_pos is not None and goal_pos is not None:
            self._render_start_goal_pos(ax, start_pos, goal_pos)

        legend_elements = [
            Line2D([0], [0], color=self.COLORS["trajectory_free"], lw=self.SIZES["trajectory_line_width"], label="Свободная траектория"),
            Line2D([0], [0], color=self.COLORS["trajectory_collision"], lw=self.SIZES["trajectory_line_width"], label="Траектория с коллизиями"),
        ]
        if best_traj_idx is not None:
            legend_elements.append(Line2D([0], [0], color=self.COLORS["trajectory_best"], lw=self.SIZES["trajectory_best_line_width"], label="Лучшая траектория"))
            
        ax.legend(handles=legend_elements, loc="upper right")

        if save_path is not None:
            fig.savefig(save_path, dpi=600, bbox_inches="tight")
            plt.close(fig)

        return fig, ax

    def _save_animation(self, fig, update_fn, n_frames, anim_time, save_path):
        animation = FuncAnimation(
            fig,
            update_fn,
            frames=n_frames,
            interval=anim_time * 1000 / n_frames,
            repeat=False,
        )

        fps = max(1, int(n_frames / anim_time))
        writer = FFMpegWriter(
            fps=fps, codec="libx264", extra_args=["-preset", "ultrafast", "-crf", "23"]
        )
        animation.save(save_path, writer=writer, dpi=90)

    def animate_robot_motion(
        self,
        trajectories: torch.Tensor,
        best_traj_idx: Optional[int] = None,
        start_pos: Optional[torch.Tensor] = None,
        goal_pos: Optional[torch.Tensor] = None,
        n_frames: int = 60,
        anim_time: int = 5,
        draw_indices: Optional[List[int]] = None,
        draw_spacing: int = 1,
        save_path: str = "robot_motion_animation.mp4",
    ) -> None:
        n_trajectories, n_support_points, n_dim = trajectories.shape
        frame_indices = np.round(np.linspace(0, n_support_points - 1, n_frames)).astype(
            int
        )
        _, _, points_collision_mask = self.robot.get_trajectories_collision_and_free(
            env=self.env, trajectories=trajectories, on_extra=self.use_extra_objects
        )

        fig, ax = plt.subplots()

        self.render_scene(
            fig=fig,
            ax=ax,
            trajectories=trajectories,
            best_traj_idx=best_traj_idx,
            start_pos=start_pos,
            goal_pos=goal_pos,
            points_collision_mask=points_collision_mask,
            save_path=None,
            draw_indices=draw_indices,
            draw_spacing=draw_spacing,
        )

        draw_indices = (
            draw_indices if draw_indices is not None else list(range(n_trajectories))
        )
        if best_traj_idx is not None and best_traj_idx in draw_indices:
            best_traj_idx = draw_indices.index(best_traj_idx)

        if best_traj_idx not in draw_indices:
            best_traj_idx = None

        n_draw = len(draw_indices)

        moving_artists = []

        dummy_collision_points = self.robot.get_collision_points(trajectories[0, 0])
        n_collision_points = dummy_collision_points.shape[-2]

        diams = np.full((n_draw * n_collision_points, 2), self.robot.margin * 2)
        spheres_collection = EllipseCollection(
            widths=diams[:, 0],
            heights=diams[:, 1],
            angles=np.zeros(n_draw * n_collision_points),
            units="xy",
            offsets=np.zeros((n_draw * n_collision_points, 2)),
            transOffset=ax.transData,
            facecolors=[self.COLORS["robot_free_moving"]]
            * (n_draw * n_collision_points),
            zorder=20,
        )
        ax.add_collection(spheres_collection)
        moving_artists.append(spheres_collection)

        if self.robot.name == "L2D":
            segments_collection = LineCollection(
                [],
                colors=[self.COLORS["robot_free_moving"]] * (n_draw * 2),
                linewidths=self.SIZES["robot_line_width_moving"],
                zorder=20,
            )
            ax.add_collection(segments_collection)
            moving_artists.append(segments_collection)

        def update_frame(frame_idx) -> List[plt.Artist]:
            idx = frame_indices[frame_idx]
            ax.set_title(f"Step: {idx}/{n_support_points - 1}")
            current_states = trajectories[draw_indices, idx, :]

            mask = points_collision_mask[draw_indices, idx].unsqueeze(1)

            colors = self._compute_robot_colors(
                mask, best_traj_idx=best_traj_idx, type="moving"
            )

            collision_points = self.robot.get_collision_points(current_states).reshape(
                -1, 2
            )
            collision_points_np = collision_points.cpu().numpy()
            spheres_collection.set_offsets(collision_points_np)

            expanded_colors = [c[0] for c in colors for _ in range(n_collision_points)]
            spheres_collection.set_facecolor(expanded_colors)

            if self.robot.name == "L2D":
                current_states_np = current_states.cpu().numpy()
                rights, bases, tops = (
                    current_states_np[:, :2],
                    current_states_np[:, 2:4],
                    current_states_np[:, 4:],
                )

                new_segments = []
                seg_colors = []
                for i in range(n_draw):
                    new_segments.append([bases[i], rights[i]])
                    new_segments.append([bases[i], tops[i]])
                    seg_colors.extend([colors[i][0], colors[i][0]])

                segments_collection.set_segments(new_segments)
                segments_collection.set_colors(seg_colors)

            return moving_artists + [ax.title]

        self._save_animation(fig, update_frame, n_frames, anim_time, save_path)

    def animate_optimization_iterations(
        self,
        trajectories: torch.Tensor,
        best_traj_idx: Optional[int] = None,
        start_pos: Optional[torch.Tensor] = None,
        goal_pos: Optional[torch.Tensor] = None,
        n_frames: int = 60,
        anim_time: int = 5,
        draw_indices: Optional[List[int]] = None,
        draw_spacing: int = 1,
        save_path: str = "trajectories_optimization_animation.mp4",
    ) -> None:
        n_iterations, n_trajectories, n_support_points, n_dim = trajectories.shape
        frame_indices = np.round(np.linspace(0, n_iterations - 1, n_frames)).astype(int)

        draw_indices = (
            draw_indices if draw_indices is not None else list(range(n_trajectories))
        )

        _, _, points_collision_mask = self.robot.get_trajectories_collision_and_free(
            env=self.env,
            trajectories=trajectories.reshape(-1, n_trajectories, n_support_points),
            on_extra=self.use_extra_objects,
        )
        points_collision_mask = points_collision_mask.reshape(
            n_iterations, n_trajectories, -1
        )

        fig, ax = plt.subplots()

        def update_frame(frame_idx):
            idx = frame_indices[frame_idx]
            ax.clear()
            ax.set_title(f"Iteration: {frame_indices[frame_idx]}/{n_iterations - 1}")

            self.render_scene(
                fig=fig,
                ax=ax,
                trajectories=trajectories[idx],
                best_traj_idx=best_traj_idx if frame_idx == n_frames - 1 else None,
                start_pos=start_pos,
                goal_pos=goal_pos,
                points_collision_mask=points_collision_mask[idx],
                draw_indices=draw_indices,
                draw_spacing=draw_spacing,
                save_path=None,
            )

            self._render_start_goal_pos(ax, start_pos, goal_pos)

        self._save_animation(fig, update_frame, n_frames, anim_time, save_path)
