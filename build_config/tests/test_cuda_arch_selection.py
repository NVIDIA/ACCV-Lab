import unittest
from unittest import mock

from accvlab_build_config.helpers import build_utils
from accvlab_build_config.helpers import cmake_args


class CudaArchSelectionTest(unittest.TestCase):
    def _mock_supported_architectures(self, supported_architectures):
        return mock.patch.object(
            build_utils,
            "_detect_nvcc_supported_architectures",
            return_value=supported_architectures,
        )

    def test_exact_supported_architecture_uses_real_target(self):
        with self._mock_supported_architectures(["80", "90", "100"]):
            selection = build_utils.select_cuda_architectures_for_nvcc(["90"])

        self.assertEqual(selection.architectures, ["90"])
        self.assertEqual(selection.ptx_architectures, [])

    def test_unsupported_hole_uses_base_ptx_not_nearby_real(self):
        with self._mock_supported_architectures(["100", "120"]):
            selection = build_utils.select_cuda_architectures_for_nvcc(["103"])

        self.assertEqual(selection.architectures, [])
        self.assertEqual(selection.ptx_architectures, ["100"])

    def test_future_gpu_uses_supported_base_ptx_below_detection(self):
        with self._mock_supported_architectures(["80", "90", "100", "103", "120"]):
            selection = build_utils.select_cuda_architectures_for_nvcc(["121"])

        self.assertEqual(selection.architectures, [])
        self.assertEqual(selection.ptx_architectures, ["120"])

    def test_unsupported_detection_uses_greatest_supported_ptx_without_base(self):
        with self._mock_supported_architectures(["75", "86", "89"]):
            selection = build_utils.select_cuda_architectures_for_nvcc(["88"])

        self.assertEqual(selection.architectures, [])
        self.assertEqual(selection.ptx_architectures, ["86"])

    def test_unsupported_detection_without_lower_support_remains_unchanged(self):
        with self._mock_supported_architectures(["60", "70"]):
            selection = build_utils.select_cuda_architectures_for_nvcc(["50"])

        self.assertEqual(selection.architectures, ["50"])
        self.assertEqual(selection.ptx_architectures, [])

    def test_mixed_exact_and_unsupported_architectures_preserve_order(self):
        with self._mock_supported_architectures(["80", "90", "100", "120"]):
            selection = build_utils.select_cuda_architectures_for_nvcc(["90", "103"])

        self.assertEqual(selection.architectures, ["90"])
        self.assertEqual(selection.ptx_architectures, ["100"])

    def test_duplicate_ptx_fallbacks_are_deduplicated(self):
        with self._mock_supported_architectures(["100", "120"]):
            selection = build_utils.select_cuda_architectures_for_nvcc(["103", "103"])

        self.assertEqual(selection.architectures, [])
        self.assertEqual(selection.ptx_architectures, ["100"])

    def test_no_detected_nvcc_architectures_returns_input_unchanged(self):
        with self._mock_supported_architectures([]):
            selection = build_utils.select_cuda_architectures_for_nvcc(["103"])

        self.assertEqual(selection.architectures, ["103"])
        self.assertEqual(selection.ptx_architectures, [])

    def test_format_torch_cuda_arch_list_converts_compact_numbers(self):
        self.assertEqual(build_utils._format_torch_cuda_arch_list(["90"]), "9.0")
        self.assertEqual(build_utils._format_torch_cuda_arch_list(["103"]), "10.3")
        self.assertEqual(build_utils._format_torch_cuda_arch_list(["120a"]), "12.0a")
        self.assertEqual(
            build_utils._format_torch_cuda_arch_list(["90", "103"]),
            "9.0;10.3",
        )

    def test_resolve_torch_cuda_arch_list_uses_custom_cuda_archs(self):
        with mock.patch.dict(
            "os.environ",
            {"CUSTOM_CUDA_ARCHS": "103"},
            clear=False,
        ):
            resolved = build_utils._resolve_torch_cuda_arch_list(
                {"cuda_available": True, "gpu_architectures": ["90"]}
            )
        self.assertEqual(resolved, "10.3")

    def test_resolve_cuda_architecture_selection_uses_custom_cuda_archs(self):
        with mock.patch.dict(
            "os.environ",
            {"CUSTOM_CUDA_ARCHS": "70,75,80"},
            clear=False,
        ):
            with mock.patch.object(
                build_utils,
                "select_cuda_architectures_for_nvcc",
                side_effect=AssertionError("unexpected selector call"),
            ):
                selection = build_utils.resolve_cuda_architecture_selection(
                    {"cuda_available": True, "gpu_architectures": ["90"]}
                )
        self.assertEqual(selection, build_utils.CudaArchitectureSelection(["70", "75", "80"], []))

    def test_build_cmake_args_custom_cuda_archs_cmake_strategy(self):
        cuda_info = {"cuda_available": True, "gpu_architectures": ["90"]}
        with mock.patch.dict("os.environ", {"CUSTOM_CUDA_ARCHS": "70,75,80"}, clear=False):
            with mock.patch.object(cmake_args, "detect_cuda_info", return_value=cuda_info):
                with mock.patch.object(
                    build_utils,
                    "select_cuda_architectures_for_nvcc",
                    side_effect=AssertionError("unexpected selector call"),
                ):
                    with mock.patch.object(
                        cmake_args,
                        "_build_cmake_args_package_scm_version",
                        return_value=[],
                    ):
                        args = cmake_args.build_cmake_args(
                            cuda_arch_strategy=cmake_args.CUDA_ARCH_STRATEGY_CMAKE
                        )

        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=70;75;80", "\n".join(args))

    def test_build_cmake_args_custom_cuda_archs_torch_strategy(self):
        cuda_info = {"cuda_available": True, "gpu_architectures": ["90"]}
        with mock.patch.dict("os.environ", {"CUSTOM_CUDA_ARCHS": "70,75,80"}, clear=False):
            with mock.patch.object(cmake_args, "detect_cuda_info", return_value=cuda_info):
                with mock.patch.object(
                    build_utils,
                    "select_cuda_architectures_for_nvcc",
                    side_effect=AssertionError("unexpected selector call"),
                ):
                    with mock.patch.object(
                        cmake_args,
                        "_build_cmake_args_package_scm_version",
                        return_value=[],
                    ):
                        args = cmake_args.build_cmake_args(
                            cuda_arch_strategy=cmake_args.CUDA_ARCH_STRATEGY_TORCH
                        )

        self.assertIn("-DACCVLAB_TORCH_CUDA_ARCH_LIST=7.0;7.5;8.0", "\n".join(args))

    def test_resolve_torch_cuda_arch_list_uses_detected_real_arch(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self._mock_supported_architectures(["80", "90", "100"]):
                resolved = build_utils._resolve_torch_cuda_arch_list(
                    {"cuda_available": True, "gpu_architectures": ["90"]}
                )
        self.assertEqual(resolved, "9.0")

    def test_resolve_torch_cuda_arch_list_adds_ptx_for_unsupported_arch(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self._mock_supported_architectures(["100", "120"]):
                resolved = build_utils._resolve_torch_cuda_arch_list(
                    {"cuda_available": True, "gpu_architectures": ["103"]}
                )
        self.assertEqual(resolved, "10.0+PTX")

    def test_build_cmake_args_cmake_strategy_emits_cmake_architectures(self):
        cuda_info = {"cuda_available": True, "gpu_architectures": ["90"]}
        with mock.patch.object(cmake_args, "detect_cuda_info", return_value=cuda_info):
            with self._mock_supported_architectures(["80", "90", "100"]):
                with mock.patch.object(
                    cmake_args,
                    "_build_cmake_args_package_scm_version",
                    return_value=[],
                ):
                    args = cmake_args.build_cmake_args(cuda_arch_strategy=cmake_args.CUDA_ARCH_STRATEGY_CMAKE)

        joined = "\n".join(args)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=90", joined)
        self.assertNotIn("-DACCVLAB_TORCH_CUDA_ARCH_LIST=", joined)

    def test_build_cmake_args_torch_strategy_emits_torch_arch_list(self):
        cuda_info = {"cuda_available": True, "gpu_architectures": ["90"]}
        with mock.patch.object(cmake_args, "detect_cuda_info", return_value=cuda_info):
            with self._mock_supported_architectures(["80", "90", "100"]):
                with mock.patch.object(
                    cmake_args,
                    "_build_cmake_args_package_scm_version",
                    return_value=[],
                ):
                    args = cmake_args.build_cmake_args(cuda_arch_strategy=cmake_args.CUDA_ARCH_STRATEGY_TORCH)

        joined = "\n".join(args)
        self.assertIn("-DACCVLAB_TORCH_CUDA_ARCH_LIST=9.0", joined)
        self.assertNotIn("-DCMAKE_CUDA_ARCHITECTURES=", joined)

    def test_explicit_custom_architectures_are_not_rewritten(self):
        with mock.patch.object(
            build_utils,
            "select_cuda_architectures_for_nvcc",
            side_effect=AssertionError("unexpected selector call"),
        ):
            config = {
                "CPP_STANDARD": "c++17",
                "OPTIMIZE_LEVEL": 3,
                "USE_FAST_MATH": False,
                "DEBUG_BUILD": False,
                "ENABLE_PROFILING": False,
                "VERBOSE_BUILD": False,
                "CUSTOM_CUDA_ARCHS": ["103"],
            }
            cuda_info = {"cuda_available": True, "gpu_architectures": ["103"]}

            flags = build_utils.get_compile_flags(config, cuda_info)

        self.assertIn("-gencode=arch=compute_103,code=sm_103", flags["nvcc"])
        self.assertNotIn("-gencode=arch=compute_100,code=compute_100", flags["nvcc"])


if __name__ == "__main__":
    unittest.main()
