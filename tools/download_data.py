#!/usr/bin/env python3
"""
Download multiple batches of 400k matches and combine with deduplication
"""

import requests
import json
import time
import os
import glob
from datetime import datetime

def download_batch(batch_num, offset, limit=400000):
    """Download a single batch with error handling"""
    
    sql = f"""SELECT * FROM public_matches 
              ORDER BY match_id DESC 
              LIMIT {limit} 
              OFFSET {offset}"""
    
    url = "https://api.opendota.com/api/explorer"
    params = {"sql": sql}
    
    print(f"📥 Downloading batch {batch_num}...")
    print(f"   Matches: {offset:,} to {offset + limit:,}")
    print(f"   Query: LIMIT {limit} OFFSET {offset}")
    
    try:
        print(f"   🌐 Making API request...")
        response = requests.get(url, params=params, timeout=120)  # 2 min timeout
        response.raise_for_status()
        
        data = response.json()
        
        # Check if we got data
        if not data.get('rows'):
            print(f"   ⚠️  No data returned (probably reached end of dataset)")
            return 0
        
        filename = f"../data/matches_batch_{batch_num}_offset_{offset//1000}k.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f)
        
        match_count = len(data.get('rows', []))
        print(f"   ✅ Success: {match_count:,} matches saved to {filename}")
        
        # Show some stats about this batch
        if match_count > 0:
            matches = data['rows']
            start_times = [m['start_time'] for m in matches if m.get('start_time')]
            if start_times:
                from datetime import datetime
                earliest = datetime.fromtimestamp(min(start_times))
                latest = datetime.fromtimestamp(max(start_times))
                print(f"   📅 Date range: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
        
        return match_count
        
    except requests.exceptions.Timeout:
        print(f"   ❌ Request timed out after 2 minutes")
        return 0
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request error: {e}")
        return 0
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return 0

def download_multiple_batches(num_batches=5, batch_size=400000, start_offset=0):
    """Download multiple batches automatically"""
    print("🚀 Starting automated batch download...")
    print(f"📊 Plan: {num_batches} batches of {batch_size:,} matches each")
    print(f"🎯 Target: {num_batches * batch_size:,} total matches")
    print(f"⏰ Estimated time: {num_batches * 0.5} minutes")
    print("="*60)
    
    total_matches = 0
    successful_batches = []
    failed_batches = []
    
    for i in range(num_batches):
        batch_num = i + 1
        offset = start_offset + (i * batch_size)
        
        print(f"\n{'='*50}")
        print(f"BATCH {batch_num}/{num_batches}")
        print(f"{'='*50}")
        
        count = download_batch(batch_num, offset, batch_size)
        
        if count > 0:
            total_matches += count
            successful_batches.append(f"../data/matches_batch_{batch_num}_offset_{offset//1000}k.json")
        else:
            failed_batches.append(batch_num)
            if count == 0:
                print(f"   🛑 No more data available, stopping early")
                break
        
        # Progress update
        print(f"   📈 Progress: {total_matches:,} matches downloaded so far")
        
        # Be nice to the API (wait between batches)
        if i < num_batches - 1 and count > 0:
            wait_time = 5  # 10 seconds between batches
            print(f"   ⏳ Waiting {wait_time} seconds before next batch...")
            time.sleep(wait_time)
    
    print(f"\n🎯 DOWNLOAD COMPLETE!")
    print(f"✅ Successful batches: {len(successful_batches)}")
    print(f"❌ Failed batches: {len(failed_batches)}")
    print(f"📊 Total matches: {total_matches:,}")
    print(f"📁 Files created:")
    for f in successful_batches:
        print(f"   - {f}")
    
    return successful_batches, total_matches

def combine_and_deduplicate(batch_files=None, output_filename=None):
    """Combine multiple batch files with deduplication"""
    print("\n🔗 COMBINING BATCHES WITH DEDUPLICATION")
    print("="*60)
    
    # Find batch files if not provided
    if batch_files is None:
        batch_files = glob.glob("../data/matches_batch_*.json")
        batch_files.sort()
    
    if not batch_files:
        print("❌ No batch files found!")
        return None
    
    print(f"📁 Found {len(batch_files)} batch files:")
    for f in batch_files:
        file_size = os.path.getsize(f) / (1024 * 1024)  # MB
        print(f"   - {f} ({file_size:.1f} MB)")
    
    all_matches = []
    total_downloaded = 0
    
    # Load all matches
    for filename in batch_files:
        print(f"\n📖 Reading {filename}...")
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            batch_matches = data.get('rows', [])
            all_matches.extend(batch_matches)
            
            print(f"   ✅ Added {len(batch_matches):,} matches")
            total_downloaded += len(batch_matches)
            
        except Exception as e:
            print(f"   ❌ Error reading {filename}: {e}")
    
    print(f"\n📊 Total downloaded: {total_downloaded:,} matches")
    
    # Deduplication
    print(f"\n🔍 Removing duplicates by match_id...")
    
    seen_ids = set()
    unique_matches = []
    duplicates = 0
    
    for match in all_matches:
        match_id = match.get('match_id')
        if match_id is None:
            print(f"   ⚠️  Warning: Found match without match_id")
            continue
            
        if match_id not in seen_ids:
            seen_ids.add(match_id)
            unique_matches.append(match)
        else:
            duplicates += 1
    
    print(f"   📊 Duplicates found: {duplicates:,}")
    print(f"   ✅ Unique matches: {len(unique_matches):,}")
    
    # Sort by match_id (descending, like original query)
    print(f"\n🔄 Sorting matches by match_id...")
    unique_matches.sort(key=lambda x: x.get('match_id', 0), reverse=True)
    
    # Date range analysis
    if unique_matches:
        start_times = [m['start_time'] for m in unique_matches if m.get('start_time')]
        if start_times:
            earliest = datetime.fromtimestamp(min(start_times))
            latest = datetime.fromtimestamp(max(start_times))
            print(f"   📅 Final date range: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
            
            # Calculate time span
            time_span = (max(start_times) - min(start_times)) / (24 * 3600)  # days
            print(f"   ⏱️  Time span: {time_span:.1f} days")
            print(f"   📈 Average: {len(unique_matches) / time_span:.0f} matches/day")
    
    # Create combined dataset
    combined_data = {
        "command": "Combined from multiple batches with deduplication",
        "rowCount": len(unique_matches),
        "rows": unique_matches,
        "metadata": {
            "processing_time": datetime.now().isoformat(),
            "source_batches": len(batch_files),
            "total_downloaded": total_downloaded,
            "duplicates_removed": duplicates,
            "final_unique_matches": len(unique_matches),
            "deduplication_rate": f"{duplicates/total_downloaded*100:.2f}%" if total_downloaded > 0 else "0%"
        }
    }
    
    # Generate output filename
    if output_filename is None:
        unique_count_k = len(unique_matches) // 1000
        output_filename = f"../data/public_matches_combined_{unique_count_k}k.json"
    
    # Save combined file
    print(f"\n💾 Saving combined dataset...")
    with open(output_filename, 'w') as f:
        json.dump(combined_data, f)
    
    file_size = os.path.getsize(output_filename) / (1024 * 1024)  # MB
    
    print(f"\n✅ COMBINATION COMPLETE!")
    print(f"📁 Output file: {output_filename} ({file_size:.1f} MB)")
    print(f"📊 Final stats:")
    print(f"   - Original downloads: {total_downloaded:,} matches")
    print(f"   - Duplicates removed: {duplicates:,} ({duplicates/total_downloaded*100:.2f}%)")
    print(f"   - Final unique matches: {len(unique_matches):,}")
    print(f"   - Data efficiency: {len(unique_matches)/total_downloaded*100:.1f}%")
    
    return output_filename

def main():
    """Main execution function"""
    print("🎮 DOTA 2 BATCH DOWNLOADER")
    print("="*60)
    print("Configuration:")
    print("- Batch size: 400k matches")
    print("- Automatic deduplication: Yes")
    print("- API timeout: 2 minutes per batch")
    print("- Inter-batch delay: 10 seconds")
    print("="*60)
    
    # Configuration
    NUM_BATCHES = 4  # Adjust this for how many batches you want
    BATCH_SIZE = 400000
    START_OFFSET = 0  # Start after your existing 500k matches
    
    print(f"\n🎯 DOWNLOAD PLAN:")
    print(f"Starting offset: {START_OFFSET:,} (after your existing data)")
    print(f"Batch size: {BATCH_SIZE:,} matches each")
    print(f"Number of batches: {NUM_BATCHES}")
    print(f"Target total: {NUM_BATCHES * BATCH_SIZE:,} new matches")
    print(f"Combined with existing: {START_OFFSET + NUM_BATCHES * BATCH_SIZE:,} total matches")
    
    response = input(f"\n🤔 Proceed with download? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    # Step 1: Download batches
    successful_files, total_matches = download_multiple_batches(
        num_batches=NUM_BATCHES,
        batch_size=BATCH_SIZE, 
        start_offset=START_OFFSET
    )
    
    if not successful_files:
        print("❌ No successful downloads, exiting.")
        return
    
    # Step 2: Combine and deduplicate
    print(f"\n🔄 Proceeding to combination phase...")
    combined_file = combine_and_deduplicate(successful_files)
    
    if combined_file:
        print(f"\n🎉 SUCCESS! Your expanded dataset is ready:")
        print(f"📁 File: {combined_file}")
        print(f"\n🚀 Next steps:")
        print(f"1. Update your ML pipeline to use {combined_file}")
        print(f"2. Retrain your hero baseline model")
        print(f"3. Compare accuracy with your 55.45% baseline")
        print(f"4. If improved, try adding synergy features!")

if __name__ == "__main__":
    main()