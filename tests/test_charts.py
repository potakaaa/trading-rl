from src.viz.sample_days import generate_samples


def test_sample_charts_render(settings, tmp_path):
    paths = generate_samples(settings, n=3, seed=2, synthetic=True, out_dir=tmp_path)
    assert len(paths) == 3 and all(p.exists() for p in paths)
    assert (tmp_path / "index.html").exists()
    assert "SYNTHETIC" in paths[0].read_text()[:200000]
