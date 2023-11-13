# Abstract Pod 🎧 - Japanese 

Generate an audio reading of the title and summary of the paper using `espnet2`. It can be used like a podcast!

Run next command.

```
python src/main.py [--config CONFIG]
```

Resources:
- Settings file: `config.ini`
- Environmental settings: `.devcontainer/Dockerfile`
- Template (handlebars) settings: `template/*.handlebars`
- CSV file containing the original text of the audio: `data/sample.csv`
