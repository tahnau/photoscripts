Personal photo organizer setup. It may work for your environment, or it may not—no warranty is provided.

Crontab

```
0  2 * * * /media/my_images/000_EXECUTE/phone_move_images.sh  /media/my_images/ios_photobackup_folder/ /media/my_images/000_INPUT && /usr/bin/python3 /media/halko/kuvat/000_EXECUTE/organize_everything.py
30 2 * * * /usr/bin/python3 find_duplicates.py /media/my_images/000_OUTPUT --remove
```
