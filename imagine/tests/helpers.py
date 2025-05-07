import os

fixture_path = os.path.realpath(
    f"{os.path.dirname(os.path.realpath(__file__))}/file_fixtures"
)
update_fixtures = os.environ.get("UPDATE_FILE_FIXTURES", False)


def assert_text_matches_fixture(
    fixture: str,
    expected_text: str,
) -> None:
    """
    Compare the content of a text with a fixture file.

    Args:
        fixture_path (str): Path to the fixture file.
        text (str): Text to compare with the fixture.

    Returns:
        bool: True if the text matches the fixture, False otherwise.
    """
    fixture_file_path = f"{fixture_path}/{fixture}.txt"

    if update_fixtures:
        with open(fixture_file_path, "w", encoding="utf-8") as f:
            f.write(expected_text)

    with open(fixture_file_path, "r") as f:
        fixture_content = f.read()

    assert expected_text == fixture_content
