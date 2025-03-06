# textarena-hackathon

## Getting Started
1. Install the latest version of Python with `brew install python` (install Brew if haven't already)
2. Create virtual env in repo folder with `python3 -m venv venv`
3. Activate venv with `. venv/bin/activate`
4. Install packages with `pip install -r requirements.txt`
5. Create `.env` file with [OPENROUTER_API_KEY](https://openrouter.ai/docs/api-reference/authentication)
6. Run `python3 play.py` to run the game

## Keeping Packages Updated
If installed any new packages, run `pip freeze > requirements.txt` to save latest packages before committing, so others can easily install them.
