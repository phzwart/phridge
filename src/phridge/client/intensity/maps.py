"""Map calculation and export mixin for IntensityModel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

try:
    from cctbx.array_family import flex
except ImportError:
    flex = None

from phridge.client.intensity.math_utils import _fom_rice_woolfson
from phridge.sfcalc.targets import Observations


class IntensityMapMixin:
    """Provides Fourier map synthesis and file export (MTZ and CCP4)."""

    def compute_map_coefficients(
        self: Any,
        weighted: bool = True,
        from_gradient: bool = True,
    ) -> Tuple[Any, Any, Any]:
        """Synthesize gradient difference map and sigmaA-weighted 2mFo-DFc and mFo-DFc maps.

        Parameters
        ----------
        weighted : bool, default True
            If True, weight the raw likelihood gradient by
            N_work * epsilon * Sigma_wilson * (1 - sigma_A^2).
            Centric reflections are scaled by -1.0 and acentric by -0.5 to place them
            on the identical physical Fourier difference scale.
            If False, returns raw gradient (-0.5 * dQ/dF_model).
        from_gradient : bool, default True
            If True, derives difference map (FOFCWT) and 2Fo-Fc map (2FOFCWT) directly
            from the likelihood gradient:
                F_diff = F_grad
                F_2fofc = D * F_model + 2 * F_grad
            This replaces the naive sqrt(max(Io, 0)) stand-in with the exact Bayesian
            conditional expectation under the likelihood target, properly handling
            negative/weak intensities and avoiding artificial noise baseline elevation.
            If False, uses the legacy/naive formula with sqrt(max(Io, 0)) and Rice-Woolfson FOM.

        Returns
        -------
        map_coeffs_grad : cctbx.miller.array
            Fourier coefficients for the likelihood gradient map (FGRAD).
        map_coeffs_2fofc : cctbx.miller.array
            Fourier coefficients for 2mFo-DFc map (2FOFCWT).
        map_coeffs_fofc : cctbx.miller.array
            Fourier coefficients for mFo-DFc map (FOFCWT).
        """
        if self.d_target_d_f_model is None or self.sigma_a_per_refl is None:
            self.compute_target_and_gradients()

        eps_np = np.asarray(self.i_obs.epsilons().data().as_double(), dtype=np.float64)
        sw_np = np.asarray(self.mean_i_per_refl, dtype=np.float64)
        sa_np = np.asarray(self.sigma_a_per_refl, dtype=np.float64)
        cen_np = np.asarray(self.i_obs.centric_flags().data(), dtype=bool)

        # 1. Gradient difference map:
        if weighted:
            n_work = float(self.i_obs.size() - self.r_free_flags.data().count(True))
            sigma_wilson = eps_np * sw_np
            var_weight = n_work * sigma_wilson * (1.0 - sa_np**2)
            factor = np.where(cen_np, -1.0, -0.5)
            grad_f = factor * self.d_target_d_f_model * var_weight
        else:
            grad_f = -0.5 * self.d_target_d_f_model

        grad_arr = np.ascontiguousarray(grad_f, dtype=np.complex128)
        try:
            grad_flex = flex.complex_double(grad_arr)
        except Exception:
            grad_flex = flex.complex_double(grad_f.tolist())

        map_coeffs_grad = self.i_obs.customized_copy(
            data=grad_flex,
            sigmas=None,
            observation_type=None,
        )

        fmod_np = np.asarray(self.f_model.data(), dtype=np.complex128)

        # 2. Difference and 2Fo-Fc maps
        if from_gradient:
            f_fofc = grad_f
            f_2fofc = sa_np * fmod_np + 2.0 * grad_f
        else:
            phi_calc = np.angle(fmod_np)
            fmod_abs = np.abs(fmod_np)
            io_np = np.asarray(self.i_obs.data(), dtype=np.float64)
            fo_amp = np.sqrt(np.maximum(io_np, 0.0))
            denom = np.maximum(eps_np * sw_np, 1e-12)
            Ec = fmod_abs / np.sqrt(denom)
            Zo = io_np / denom
            m = _fom_rice_woolfson(Ec, sa_np, Zo, cen_np)
            D = sa_np
            phase_factor = np.exp(1j * phi_calc)
            f_2fofc = (2.0 * m * fo_amp - D * fmod_abs) * phase_factor
            f_fofc = (m * fo_amp - D * fmod_abs) * phase_factor

        f_2fofc_arr = np.ascontiguousarray(f_2fofc, dtype=np.complex128)
        f_fofc_arr = np.ascontiguousarray(f_fofc, dtype=np.complex128)
        try:
            c_2fofc_flex = flex.complex_double(f_2fofc_arr)
            c_fofc_flex = flex.complex_double(f_fofc_arr)
        except Exception:
            c_2fofc_flex = flex.complex_double(f_2fofc.tolist())
            c_fofc_flex = flex.complex_double(f_fofc.tolist())

        map_coeffs_2fofc = self.i_obs.customized_copy(
            data=c_2fofc_flex,
            sigmas=None,
            observation_type=None,
        )
        map_coeffs_fofc = self.i_obs.customized_copy(
            data=c_fofc_flex,
            sigmas=None,
            observation_type=None,
        )

        return map_coeffs_grad, map_coeffs_2fofc, map_coeffs_fofc

    def compute_all_map_coefficients(
        self: Any,
        newton_damping: float = 0.1,
        include_free: bool = True,
    ) -> Dict[str, Any]:
        """Compute all map coefficients and per-reflection posterior statistics."""
        if self.sigma_a_per_refl is None:
            self.estimate_sigma_a()
        if self.d_target_d_f_model is None:
            self.compute_target_and_gradients()

        grad_coeffs, coeffs_2fofc, coeffs_fofc = self.compute_map_coefficients(
            weighted=True, from_gradient=True
        )
        raw_grad_coeffs, _, _ = self.compute_map_coefficients(
            weighted=False, from_gradient=True
        )

        res: Dict[str, Any] = {
            "gradient_weighted": grad_coeffs,
            "gradient_raw": raw_grad_coeffs,
            "2fofc": coeffs_2fofc,
            "fofc": coeffs_fofc,
        }

        if self.target_type != "intensity":
            return res

        from phridge.contrib.intensity_ll.maps import (
            IntensityMapOptions,
            intensity_map_coefficients,
        )
        from phridge.contrib.intensity_ll.target import IntensityLogLikelihood

        io_np = np.asarray(self.i_obs.data(), dtype=np.float64)
        si_np = np.asarray(self.i_obs.sigmas(), dtype=np.float64)
        eps_np = np.asarray(self.i_obs.epsilons().data().as_double(), dtype=np.float64)
        cen_np = np.asarray(self.i_obs.centric_flags().data(), dtype=bool)
        sa_np = np.asarray(self.sigma_a_per_refl, dtype=np.float64)
        sw_np = np.asarray(self.mean_i_per_refl, dtype=np.float64)
        r_free_np = np.asarray(self.r_free_flags.data(), dtype=bool)

        nu_np = None
        if self.nu_per_refl is not None:
            nu_np = np.asarray(self.nu_per_refl, dtype=np.float64)
        elif self.nu is not None:
            nu_np = np.full(self.i_obs.size(), self.nu, dtype=np.float64)

        obs = Observations.from_numpy(
            device=self.device,
            dtype=self.float_dtype,
            data=io_np,
            sigmas=si_np,
            epsilon=eps_np,
            centric=cen_np,
            alpha=sa_np,
            beta=sw_np,
            r_free=r_free_np,
            nu=nu_np,
        )
        tgt_kwargs: Dict[str, Any] = {}
        if self.nu is not None:
            tgt_kwargs["nu"] = float(self.nu)
        tgt = IntensityLogLikelihood(**tgt_kwargs)

        fmod_np = np.asarray(self.f_model.data(), dtype=np.complex128)
        fmod_t = torch.as_tensor(fmod_np, dtype=self.complex_dtype, device=self.torch_device)
        opts = IntensityMapOptions(
            newton_damping=newton_damping,
            include_free=include_free,
        )
        ms = intensity_map_coefficients(tgt, fmod_t, obs, opts)

        def _to_miller(c_data: np.ndarray) -> Any:
            c_arr = np.ascontiguousarray(c_data, dtype=np.complex128)
            try:
                data = flex.complex_double(c_arr)
            except Exception:
                data = flex.complex_double(c_data.tolist())
            return self.i_obs.customized_copy(
                data=data,
                sigmas=None,
                observation_type=None,
            )

        def _to_real_miller(r_data: np.ndarray) -> Any:
            r_arr = np.ascontiguousarray(r_data, dtype=np.float64)
            try:
                data = flex.double(r_arr)
            except Exception:
                data = flex.double(r_data.tolist())
            return self.i_obs.customized_copy(
                data=data,
                sigmas=None,
                observation_type=None,
            )

        res.update(
            difference=_to_miller(ms.difference),
            model=_to_miller(ms.model),
            gradient=_to_miller(ms.gradient),
            newton=_to_miller(ms.newton),
            fom=_to_real_miller(ms.fom),
            f_post=_to_real_miller(ms.f_post),
            robust_weight=_to_real_miller(ms.robust_weight),
            curvature=_to_real_miller(ms.curvature),
            stats=ms.stats,
        )
        if ms.d_loglik_d_nu is not None:
            res["d_loglik_d_nu"] = _to_real_miller(ms.d_loglik_d_nu)

        return res

    def write_maps(
        self: Any,
        prefix: str = "output",
        resolution_factor: float = 0.25,
        write_ccp4: bool = True,
        write_mtz: bool = True,
        weighted_gradient: bool = True,
        from_gradient: bool = True,
        compute_all: bool = True,
    ) -> Dict[str, str]:
        """Export synthesized maps to MTZ and real-space CCP4 (.ccp4) formats."""
        grad_coeffs, coeffs_2fofc, coeffs_fofc = self.compute_map_coefficients(
            weighted=weighted_gradient,
            from_gradient=from_gradient,
        )
        raw_grad_coeffs, _, _ = self.compute_map_coefficients(
            weighted=False,
            from_gradient=from_gradient,
        )
        all_maps: Dict[str, Any] = {}
        if compute_all and self.target_type == "intensity":
            try:
                all_maps = self.compute_all_map_coefficients()
            except Exception as e:
                import warnings

                warnings.warn(f"Could not compute all posterior map coefficients: {e}")

        written_files = {}

        # Write MTZ
        if write_mtz:
            mtz_path = f"{prefix}_maps.mtz"
            mtz_dataset = coeffs_2fofc.as_mtz_dataset(column_root_label="2FOFCWT")
            mtz_dataset.add_miller_array(coeffs_fofc, column_root_label="FOFCWT")
            mtz_dataset.add_miller_array(grad_coeffs, column_root_label="FGRAD")
            mtz_dataset.add_miller_array(raw_grad_coeffs, column_root_label="FGRAD_RAW")
            if "newton" in all_maps:
                mtz_dataset.add_miller_array(all_maps["newton"], column_root_label="FNEWTON")
            if "difference" in all_maps:
                mtz_dataset.add_miller_array(all_maps["difference"], column_root_label="FDIFF_POST")
            if "model" in all_maps:
                mtz_dataset.add_miller_array(all_maps["model"], column_root_label="FMODEL_POST")
            if "fom" in all_maps:
                mtz_dataset.add_miller_array(all_maps["fom"], column_root_label="FOM")
            if "f_post" in all_maps:
                mtz_dataset.add_miller_array(all_maps["f_post"], column_root_label="F_POST")
            if "robust_weight" in all_maps and self.nu is not None:
                mtz_dataset.add_miller_array(all_maps["robust_weight"], column_root_label="ROBUST_WT")
            mtz_dataset.mtz_object().write(mtz_path)
            written_files["mtz"] = mtz_path

        # Write CCP4 maps
        if write_ccp4:
            map_targets = [
                ("gradient", grad_coeffs),
                ("gradient_raw", raw_grad_coeffs),
                ("2fofc", coeffs_2fofc),
                ("fofc", coeffs_fofc),
            ]
            if "newton" in all_maps:
                map_targets.append(("newton", all_maps["newton"]))
            if "difference" in all_maps:
                map_targets.append(("diff_post", all_maps["difference"]))
            if "model" in all_maps:
                map_targets.append(("model_post", all_maps["model"]))
            for name, coeffs in map_targets:
                ccp4_path = f"{prefix}_{name}.ccp4"
                fft_map = coeffs.fft_map(resolution_factor=resolution_factor)
                fft_map.apply_sigma_scaling()
                fft_map.as_ccp4_map(
                    file_name=ccp4_path,
                    labels=[f"phridge {name} map"],
                )
                written_files[name] = ccp4_path

        return written_files
