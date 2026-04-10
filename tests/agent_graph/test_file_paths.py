from agent_graph.file_paths import to_local_file_path


def test_relative_agent_path_is_resolved_under_gaia_files_root():
    assert (
        to_local_file_path("2023/validation/abc.png")
        == ".gaia-questions/files/2023/validation/abc.png"
    )


def test_single_segment_path_is_resolved():
    assert to_local_file_path("foo.txt") == ".gaia-questions/files/foo.txt"


def test_absolute_path_is_returned_unchanged():
    assert to_local_file_path("/tmp/elsewhere.png") == "/tmp/elsewhere.png"


def test_path_already_under_files_root_is_not_double_prefixed():
    already = ".gaia-questions/files/2023/validation/abc.png"
    assert to_local_file_path(already) == already
