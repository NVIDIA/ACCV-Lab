def test_draw_heatmap_package_imports():
    """Regression test for misspelled submodule import in draw_heatmap.__init__."""
    import accvlab.draw_heatmap

    assert hasattr(accvlab.draw_heatmap, "draw_heatmap")
    assert hasattr(accvlab.draw_heatmap, "draw_heatmap_batched")
    assert "draw_heatmap" in accvlab.draw_heatmap.__all__
    assert "draw_heatmap_batched" in accvlab.draw_heatmap.__all__
