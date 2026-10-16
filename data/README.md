# Dataset folder

Put the dataset here. Nothing is downloaded automatically.

**Recommended: UCI SMS Spam Collection** (SMS messages, NOT emails)

1. Open https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. Click **Download** to get `sms+spam+collection.zip`
3. Copy the zip into this `data/` folder (do not rename the contents).
   The program extracts it automatically, or you can extract it yourself so that
   the file `SMSSpamCollection` is in this folder.

Format of `SMSSpamCollection`: one message per line, `label<TAB>message`, no header,
label is `ham` or `spam`.

A CSV with a header row also works (e.g. Kaggle `spam.csv` with columns `v1,v2`, or any
CSV with a label column and a text column such as `label,text`).
