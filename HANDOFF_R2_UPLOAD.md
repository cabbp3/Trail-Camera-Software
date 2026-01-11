# Handoff: Upload Thumbnails with Hash-Based Names

## YOUR TASK: Re-upload thumbnails using file_hash

The naming scheme changed from `{photo_id}_thumb.jpg` to `{file_hash}_thumb.jpg` for security.

### Step 1: Run the upload
```bash
cd "/Users/brookebratcher/Desktop/Trail Camera Software V 1.0"
source .venv/bin/activate
python3 tools/batch_upload_r2.py --username brooke --thumbnails-only --yes
```

This will:
- Upload thumbnails with hash-based filenames
- Automatically skip files that already exist (deduplication)
- Takes 10-20 minutes for ~7,000 photos

### Step 2: Verify upload completed
```bash
cd "/Users/brookebratcher/Desktop/Trail Camera Software V 1.0" && source .venv/bin/activate && python3 -c "
import boto3
import json

with open('/Users/brookebratcher/.trailcam/r2_config.json') as f:
    config = json.load(f)

s3 = boto3.client('s3', endpoint_url=config['endpoint_url'],
    aws_access_key_id=config['access_key_id'],
    aws_secret_access_key=config['secret_access_key'], region_name='auto')

# Count hash-based thumbnails
paginator = s3.get_paginator('list_objects_v2')
count = 0
for page in paginator.paginate(Bucket=config['bucket_name'], Prefix='users/brooke/thumbnails/'):
    for obj in page.get('Contents', []):
        # Hash-based files have 32-char hash before _thumb.jpg
        key = obj['Key']
        filename = key.split('/')[-1]
        if len(filename) > 40:  # hash (32) + _thumb.jpg (10)
            count += 1

print(f'Hash-based thumbnails: {count}')
"
```

**Success criteria:** Should show 7,000+ hash-based thumbnails.

### Step 3: Report back
Tell the user: "Upload complete with hash-based filenames. X thumbnails uploaded."

---

## What Changed

| Before | After |
|--------|-------|
| `12345_thumb.jpg` | `44eb84f49521a82844d1..._thumb.jpg` |
| Sequential IDs (guessable) | MD5 hashes (unguessable) |
| Presigned URLs | Public URLs |

## DO NOT DO
- Don't touch mobile app code
- Don't modify r2_service.dart or photo_service.dart
- Just run the upload and report completion
