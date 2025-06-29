#!/usr/bin/env python3
"""
Presidential Speech Downloader
Download all presidential speeches with comprehensive metadata from Miller Center API
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


class PresidentialSpeechDownloader:
    def __init__(self):
        self.api_endpoint = "https://api.millercenter.org/speeches"
        self.presidents_data = self._load_presidents_data()

    def _load_presidents_data(self) -> Dict:
        """Load comprehensive president metadata"""
        return {
            "George Washington": {
                "term": 1,
                "years": "1789-1797",
                "party": "Unaffiliated",
                "number": 1,
            },
            "John Adams": {
                "term": 1,
                "years": "1797-1801",
                "party": "Federalist",
                "number": 2,
            },
            "Thomas Jefferson": {
                "term": 2,
                "years": "1801-1809",
                "party": "Democratic-Republican",
                "number": 3,
            },
            "James Madison": {
                "term": 2,
                "years": "1809-1817",
                "party": "Democratic-Republican",
                "number": 4,
            },
            "James Monroe": {
                "term": 2,
                "years": "1817-1825",
                "party": "Democratic-Republican",
                "number": 5,
            },
            "John Quincy Adams": {
                "term": 1,
                "years": "1825-1829",
                "party": "Democratic-Republican",
                "number": 6,
            },
            "Andrew Jackson": {
                "term": 2,
                "years": "1829-1837",
                "party": "Democratic",
                "number": 7,
            },
            "Martin Van Buren": {
                "term": 1,
                "years": "1837-1841",
                "party": "Democratic",
                "number": 8,
            },
            "William Henry Harrison": {
                "term": 1,
                "years": "1841",
                "party": "Whig",
                "number": 9,
            },
            "John Tyler": {
                "term": 1,
                "years": "1841-1845",
                "party": "Whig",
                "number": 10,
            },
            "James K. Polk": {
                "term": 1,
                "years": "1845-1849",
                "party": "Democratic",
                "number": 11,
            },
            "Zachary Taylor": {
                "term": 1,
                "years": "1849-1850",
                "party": "Whig",
                "number": 12,
            },
            "Millard Fillmore": {
                "term": 1,
                "years": "1850-1853",
                "party": "Whig",
                "number": 13,
            },
            "Franklin Pierce": {
                "term": 1,
                "years": "1853-1857",
                "party": "Democratic",
                "number": 14,
            },
            "James Buchanan": {
                "term": 1,
                "years": "1857-1861",
                "party": "Democratic",
                "number": 15,
            },
            "Abraham Lincoln": {
                "term": 2,
                "years": "1861-1865",
                "party": "Republican",
                "number": 16,
            },
            "Andrew Johnson": {
                "term": 1,
                "years": "1865-1869",
                "party": "Democratic",
                "number": 17,
            },
            "Ulysses S. Grant": {
                "term": 2,
                "years": "1869-1877",
                "party": "Republican",
                "number": 18,
            },
            "Rutherford B. Hayes": {
                "term": 1,
                "years": "1877-1881",
                "party": "Republican",
                "number": 19,
            },
            "James A. Garfield": {
                "term": 1,
                "years": "1881",
                "party": "Republican",
                "number": 20,
            },
            "Chester A. Arthur": {
                "term": 1,
                "years": "1881-1885",
                "party": "Republican",
                "number": 21,
            },
            "Grover Cleveland": {
                "term": 2,
                "years": "1885-1889, 1893-1897",
                "party": "Democratic",
                "number": 22,
            },
            "Benjamin Harrison": {
                "term": 1,
                "years": "1889-1893",
                "party": "Republican",
                "number": 23,
            },
            "William McKinley": {
                "term": 2,
                "years": "1897-1901",
                "party": "Republican",
                "number": 25,
            },
            "Theodore Roosevelt": {
                "term": 2,
                "years": "1901-1909",
                "party": "Republican",
                "number": 26,
            },
            "William Howard Taft": {
                "term": 1,
                "years": "1909-1913",
                "party": "Republican",
                "number": 27,
            },
            "Woodrow Wilson": {
                "term": 2,
                "years": "1913-1921",
                "party": "Democratic",
                "number": 28,
            },
            "Warren G. Harding": {
                "term": 1,
                "years": "1921-1923",
                "party": "Republican",
                "number": 29,
            },
            "Calvin Coolidge": {
                "term": 2,
                "years": "1923-1929",
                "party": "Republican",
                "number": 30,
            },
            "Herbert Hoover": {
                "term": 1,
                "years": "1929-1933",
                "party": "Republican",
                "number": 31,
            },
            "Franklin D. Roosevelt": {
                "term": 4,
                "years": "1933-1945",
                "party": "Democratic",
                "number": 32,
            },
            "Harry S. Truman": {
                "term": 2,
                "years": "1945-1953",
                "party": "Democratic",
                "number": 33,
            },
            "Dwight D. Eisenhower": {
                "term": 2,
                "years": "1953-1961",
                "party": "Republican",
                "number": 34,
            },
            "John F. Kennedy": {
                "term": 1,
                "years": "1961-1963",
                "party": "Democratic",
                "number": 35,
            },
            "Lyndon B. Johnson": {
                "term": 2,
                "years": "1963-1969",
                "party": "Democratic",
                "number": 36,
            },
            "Richard Nixon": {
                "term": 2,
                "years": "1969-1974",
                "party": "Republican",
                "number": 37,
            },
            "Gerald Ford": {
                "term": 1,
                "years": "1974-1977",
                "party": "Republican",
                "number": 38,
            },
            "Jimmy Carter": {
                "term": 1,
                "years": "1977-1981",
                "party": "Democratic",
                "number": 39,
            },
            "Ronald Reagan": {
                "term": 2,
                "years": "1981-1989",
                "party": "Republican",
                "number": 40,
            },
            "George H. W. Bush": {
                "term": 1,
                "years": "1989-1993",
                "party": "Republican",
                "number": 41,
            },
            "Bill Clinton": {
                "term": 2,
                "years": "1993-2001",
                "party": "Democratic",
                "number": 42,
            },
            "George W. Bush": {
                "term": 2,
                "years": "2001-2009",
                "party": "Republican",
                "number": 43,
            },
            "Barack Obama": {
                "term": 2,
                "years": "2009-2017",
                "party": "Democratic",
                "number": 44,
            },
            "Donald Trump": {
                "term": 2,
                "years": "2017-2021, 2025-",
                "party": "Republican",
                "number": 45,
            },
            "Joe Biden": {
                "term": 1,
                "years": "2021-2025",
                "party": "Democratic",
                "number": 46,
            },
        }

    def download_all_speeches(self) -> List[Dict]:
        """Download all speeches from Miller Center API with pagination"""
        print("Downloading all presidential speeches from Miller Center API...")
        print("Fetching initial batch to determine total count...")

        all_speeches = []

        response = requests.post(url=self.api_endpoint)
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}")

        data = response.json()
        all_speeches.extend(data["Items"])

        # Estimate total count based on pagination pattern
        # The API doesn't give us total count, so we estimate from first batch size
        estimated_total = self._estimate_total_speeches(data)

        print(
            f"Downloaded {len(data['Items'])} speeches (estimated total: ~{estimated_total})"
        )

        # Handle pagination with progress tracking
        batch_count = 1
        while "LastEvaluatedKey" in data:
            time.sleep(0.5)  # Rate limiting

            parameters = {"LastEvaluatedKey": data["LastEvaluatedKey"]["doc_name"]}
            response = requests.post(url=self.api_endpoint, params=parameters)

            if response.status_code != 200:
                print(
                    f"Warning: Pagination request failed with status {response.status_code}"
                )
                break

            data = response.json()
            all_speeches.extend(data["Items"])
            batch_count += 1

            # Calculate progress percentage
            current_count = len(all_speeches)
            if estimated_total > 0:
                progress_percent = min((current_count / estimated_total) * 100, 100)
                print(
                    f"Downloaded {len(data['Items'])} more speeches... Total: {current_count} ({progress_percent:.1f}% complete)"
                )
            else:
                print(
                    f"Downloaded {len(data['Items'])} more speeches... Total: {current_count}"
                )

        final_count = len(all_speeches)
        print(f"\n✅ Download complete! Total speeches: {final_count}")

        # Update our estimate for future runs
        if final_count != estimated_total:
            print(
                f"   (Actual total was {final_count}, estimated was {estimated_total})"
            )

        return all_speeches

    def _estimate_total_speeches(self, first_batch_data: Dict) -> int:
        """Estimate total number of speeches based on first batch"""
        # Miller Center API typically returns batches of 40-50 speeches
        # We can estimate based on the pattern, but this is just an approximation
        first_batch_size = len(first_batch_data.get("Items", []))

        # If we have a LastEvaluatedKey, there are more batches
        if "LastEvaluatedKey" in first_batch_data:
            # Conservative estimate: assume 15-20 batches total based on historical data
            # This is just for progress indication, not critical for functionality
            estimated_batches = 18  # This gives us roughly 700-900 total speeches
            return first_batch_size * estimated_batches
        else:
            # Only one batch
            return first_batch_size

    def extract_date_from_speech(self, speech: Dict) -> Optional[Tuple[int, str]]:
        """Extract year and date from speech data"""
        # Try different date fields and formats
        date_fields = ["date", "speech_date", "delivered"]
        date_str = None

        for field in date_fields:
            if field in speech and speech[field]:
                date_str = speech[field]
                break

        if not date_str:
            # Try to extract from title or doc_name
            title = speech.get("title", "")
            doc_name = speech.get("doc_name", "")

            # Look for date patterns in title
            date_pattern = r"(\d{4})"
            match = re.search(date_pattern, title + " " + doc_name)
            if match:
                year = int(match.group(1))
                return year, f"{year}-01-01"

        if date_str:
            # Parse various date formats
            date_patterns = [
                r"(\d{4})-(\d{1,2})-(\d{1,2})",  # YYYY-MM-DD
                r"(\d{1,2})/(\d{1,2})/(\d{4})",  # MM/DD/YYYY
                r"(\w+)\s+(\d{1,2}),?\s+(\d{4})",  # Month DD, YYYY
                r"(\d{4})",  # Just year
            ]

            for pattern in date_patterns:
                match = re.search(pattern, date_str)
                if match:
                    if len(match.groups()) == 1:  # Just year
                        year = int(match.group(1))
                        return year, date_str
                    elif len(match.groups()) == 3:
                        try:
                            if pattern == date_patterns[0]:  # YYYY-MM-DD
                                year = int(match.group(1))
                            elif pattern == date_patterns[1]:  # MM/DD/YYYY
                                year = int(match.group(3))
                            else:  # Month DD, YYYY
                                year = int(match.group(3))
                            return year, date_str
                        except ValueError:
                            continue

        return None, None

    def classify_speech_type(self, speech: Dict) -> str:
        """Classify the type of speech based on title and content"""
        title = speech.get("title", "").lower()
        doc_name = speech.get("doc_name", "").lower()

        text_to_check = f"{title} {doc_name}"

        # Define speech type classifications
        if any(
            keyword in text_to_check
            for keyword in ["state of the union", "annual message", "sotu"]
        ):
            return "State of the Union"
        elif any(keyword in text_to_check for keyword in ["inaugural", "inauguration"]):
            return "Inaugural Address"
        elif any(keyword in text_to_check for keyword in ["farewell"]):
            return "Farewell Address"
        elif any(keyword in text_to_check for keyword in ["victory", "election night"]):
            return "Victory Speech"
        elif any(keyword in text_to_check for keyword in ["concession"]):
            return "Concession Speech"
        elif any(keyword in text_to_check for keyword in ["convention", "nomination"]):
            return "Convention Speech"
        elif any(keyword in text_to_check for keyword in ["congress", "joint session"]):
            return "Address to Congress"
        elif any(
            keyword in text_to_check
            for keyword in ["nation", "american people", "oval office"]
        ):
            return "Address to the Nation"
        elif any(
            keyword in text_to_check
            for keyword in ["press conference", "news conference"]
        ):
            return "Press Conference"
        elif any(keyword in text_to_check for keyword in ["campaign", "rally"]):
            return "Campaign Speech"
        elif any(keyword in text_to_check for keyword in ["commencement"]):
            return "Commencement Address"
        elif any(keyword in text_to_check for keyword in ["memorial", "remembrance"]):
            return "Memorial Address"
        elif any(keyword in text_to_check for keyword in ["proclamation"]):
            return "Proclamation"
        else:
            return "Other"

    def process_speeches(self, all_speeches: List[Dict]) -> List[Dict]:
        """Process and enhance all speeches with metadata"""
        processed_speeches = []

        for speech in all_speeches:
            year, date_str = self.extract_date_from_speech(speech)
            speech_type = self.classify_speech_type(speech)

            # Add extracted metadata
            speech["extracted_year"] = year
            speech["extracted_date"] = date_str
            speech["speech_type"] = speech_type

            processed_speeches.append(speech)

        # Sort by year and president
        processed_speeches.sort(
            key=lambda x: (x.get("extracted_year", 0), x.get("president", ""))
        )

        return processed_speeches

    def enhance_speech_metadata(self, speech: Dict) -> Dict:
        """Add comprehensive metadata to speech data"""
        president_name = speech.get("president", "")
        president_name = self._normalize_president_name(president_name)

        president_info = self.presidents_data.get(president_name, {})

        year = speech.get("extracted_year")
        date_str = speech.get("extracted_date")
        speech_type = speech.get("speech_type", "Unknown")

        enhanced_metadata = {
            "download_info": {
                "downloaded_at": datetime.now().isoformat(),
                "source": "Miller Center API",
                "api_endpoint": self.api_endpoint,
            },
            "speech_metadata": {
                "year": year,
                "date": date_str,
                "title": speech.get("title", ""),
                "doc_name": speech.get("doc_name", ""),
                "speech_type": speech_type,
                "miller_center_url": f"https://millercenter.org/the-presidency/presidential-speeches/{speech.get('doc_name', '')}",
            },
            "president_metadata": {
                "name": president_name,
                "presidential_number": president_info.get("number"),
                "party": president_info.get("party"),
                "terms_served": president_info.get("term"),
                "years_in_office": president_info.get("years"),
                "administration": f"{president_name} Administration",
            },
            "content": {
                "transcript": speech.get("transcript", speech.get("text", "")),
                "word_count": (
                    len(speech.get("transcript", speech.get("text", "")).split())
                    if speech.get("transcript") or speech.get("text")
                    else 0
                ),
            },
            "historical_context": self._get_historical_context(year, president_name),
            "classification": {
                "is_state_of_union": speech_type == "State of the Union",
                "is_inaugural": speech_type == "Inaugural Address",
                "is_farewell": speech_type == "Farewell Address",
                "is_campaign": speech_type == "Campaign Speech",
                "is_address_to_congress": speech_type == "Address to Congress",
                "is_address_to_nation": speech_type == "Address to the Nation",
            },
        }

        return enhanced_metadata

    def _normalize_president_name(self, name: str) -> str:
        """Normalize president name variations"""
        name_mappings = {
            "Franklin Delano Roosevelt": "Franklin D. Roosevelt",
            "FDR": "Franklin D. Roosevelt",
            "JFK": "John F. Kennedy",
            "LBJ": "Lyndon B. Johnson",
            "George Bush": "George H. W. Bush",
            "George W Bush": "George W. Bush",
            "Barack Hussein Obama": "Barack Obama",
            "Donald J. Trump": "Donald Trump",
            "Donald J Trump": "Donald Trump",
            "Joseph R. Biden": "Joe Biden",
            "Joe Biden Jr.": "Joe Biden",
        }

        return name_mappings.get(name, name)

    def _get_historical_context(self, year: int, president: str) -> Dict:
        """Add historical context for the speech year"""
        if not year:
            return {}

        context = {
            "decade": f"{year//10*10}s",
            "century": f"{(year-1)//100 + 1}{'st' if (year-1)//100 + 1 == 1 else 'nd' if (year-1)//100 + 1 == 2 else 'rd' if (year-1)//100 + 1 == 3 else 'th'} century",
        }

        # Add major historical periods
        if year <= 1800:
            context["period"] = "Early Republic"
        elif year <= 1860:
            context["period"] = "Antebellum Period"
        elif year <= 1865:
            context["period"] = "Civil War Era"
        elif year <= 1900:
            context["period"] = "Reconstruction & Gilded Age"
        elif year <= 1920:
            context["period"] = "Progressive Era & WWI"
        elif year <= 1940:
            context["period"] = "Interwar Period"
        elif year <= 1945:
            context["period"] = "World War II"
        elif year <= 1990:
            context["period"] = "Cold War Era"
        elif year <= 2001:
            context["period"] = "Post-Cold War"
        else:
            context["period"] = "21st Century"

        return context

    def save_speeches(
        self, speeches: List[Dict], output_dir: str = "presidential_speeches"
    ) -> str:
        """Save all speeches to individual files and create summary"""
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        saved_files = []
        summary_data = []
        speech_type_counts = {}

        for speech in speeches:
            enhanced_speech = self.enhance_speech_metadata(speech)

            # Create safe filename
            year = enhanced_speech["speech_metadata"]["year"] or "Unknown"
            president = enhanced_speech["president_metadata"]["name"] or "Unknown"
            speech_type = enhanced_speech["speech_metadata"]["speech_type"]
            title = enhanced_speech["speech_metadata"]["title"]

            safe_president = re.sub(r"[^\w\s-]", "", president).strip()
            safe_title = re.sub(r"[^\w\s-]", "", title).strip()[
                :30
            ]  # Limit title length

            filename = f"{year}_{safe_president}_{safe_title}.json"
            filename = re.sub(r"\s+", "_", filename)

            filepath = Path(output_dir) / filename

            # Save individual speech
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(enhanced_speech, f, indent=2, ensure_ascii=False)

            saved_files.append(str(filepath))

            # Count speech types
            speech_type_counts[speech_type] = speech_type_counts.get(speech_type, 0) + 1

            # Add to summary
            summary_data.append(
                {
                    "year": year,
                    "president": president,
                    "title": title,
                    "speech_type": speech_type,
                    "word_count": enhanced_speech["content"]["word_count"],
                    "party": enhanced_speech["president_metadata"]["party"],
                    "filename": filename,
                }
            )

        # Create summary file
        summary_filepath = Path(output_dir) / "presidential_speeches_summary.json"

        years_with_data = [
            s["year"] for s in summary_data if isinstance(s["year"], int)
        ]

        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_speeches": len(speeches),
            "year_range": (
                f"{min(years_with_data)} - {max(years_with_data)}"
                if years_with_data
                else "Unknown"
            ),
            "speech_type_counts": speech_type_counts,
            "presidents_represented": len(set(s["president"] for s in summary_data)),
            "speeches": summary_data,
        }

        with open(summary_filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return output_dir

    def download_all_presidential_speeches(
        self, output_dir: str = "presidential_speeches"
    ) -> str:
        """Main function to download and save all presidential speeches"""
        print("Presidential Speech Downloader")
        print("=" * 50)

        # Download all speeches
        all_speeches = self.download_all_speeches()

        # Process and enhance speeches
        print("\nProcessing and classifying speeches...")
        processed_speeches = self.process_speeches(all_speeches)

        # Count speech types
        type_counts = {}
        for speech in processed_speeches:
            speech_type = speech.get("speech_type", "Unknown")
            type_counts[speech_type] = type_counts.get(speech_type, 0) + 1

        print(f"Speech types found:")
        for speech_type, count in sorted(type_counts.items()):
            print(f"  {speech_type}: {count}")

        # Save all speeches
        print(f"\nSaving speeches to {output_dir}/...")
        output_path = self.save_speeches(processed_speeches, output_dir)

        # Print summary
        self._print_download_summary(processed_speeches, output_path, type_counts)

        return output_path

    def _print_download_summary(
        self, speeches: List[Dict], output_path: str, type_counts: Dict
    ):
        """Print a summary of downloaded speeches"""
        years = [s.get("extracted_year") for s in speeches if s.get("extracted_year")]
        presidents = list(set(s.get("president", "Unknown") for s in speeches))

        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETE!")
        print("=" * 60)
        print(f"Total speeches downloaded: {len(speeches)}")
        if years:
            print(f"Year range: {min(years)} - {max(years)}")
        print(f"Presidents represented: {len(presidents)}")
        print(f"Output directory: {output_path}")
        print(f"Summary file: {output_path}/presidential_speeches_summary.json")

        print(f"\nMost common speech types:")
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        for speech_type, count in sorted_types[:5]:
            print(f"  {speech_type}: {count}")

        print(f"\nTo find State of the Union speeches:")
        sotu_count = type_counts.get("State of the Union", 0)
        print(f"  {sotu_count} State of the Union speeches available")
        print(f"  Filter by: speech_metadata.speech_type == 'State of the Union'")
        print(f"  Or use: classification.is_state_of_union == true")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Download all presidential speeches from Miller Center"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="presidential_speeches",
        help="Output directory for downloaded speeches (default: presidential_speeches)",
    )

    args = parser.parse_args()

    downloader = PresidentialSpeechDownloader()

    try:
        output_path = downloader.download_all_presidential_speeches(args.output_dir)
        if output_path:
            print(
                f"\n✅ Successfully downloaded all presidential speeches to {output_path}/"
            )
        else:
            print(f"\n❌ No speeches were downloaded")
    except Exception as e:
        print(f"\n❌ Error downloading speeches: {e}")


if __name__ == "__main__":
    main()
