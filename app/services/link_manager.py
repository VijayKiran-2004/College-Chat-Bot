class LinkManager:
    """
    Manages topic-based website navigation links
    and extraction from documents.
    """

    TOPIC_LINKS = {
        "admission": {
            "title": "Admissions",
            "url": "https://tkrcet.ac.in/admission-procedure/",
        },
        "fee": {
            "title": "Fee Structure",
            "url": "https://tkrcet.ac.in/fee-structure/",
        },
        "placement": {
            "title": "Placements",
            "url": "https://tkrcet.ac.in/placements/",
        },
        "syllabus": {
            "title": "Syllabus",
            "url": "https://tkrcet.ac.in/syllabus/",
        },
        "principal": {
            "title": "Principal",
            "url": "https://tkrcet.ac.in/principal/",
        },
        "chairman": {
            "title": "Chairman's Message",
            "url": "https://tkrcet.ac.in/chairmans-message/",
        },
        "cse": {
            "title": "CSE Department",
            "url": (
                "https://tkrcet.ac.in/"
                "computer-science-engineering/"
            ),
        },
        "ece": {
            "title": "ECE Department",
            "url": (
                "https://tkrcet.ac.in/"
                "electronics-communication-engineering/"
            ),
        },
        "eee": {
            "title": "EEE Department",
            "url": (
                "https://tkrcet.ac.in/"
                "electrical-electronics-engineering/"
            ),
        },
        "it": {
            "title": "IT Department",
            "url": "https://tkrcet.ac.in/information-technology/",
        },
        "mech": {
            "title": "Mechanical Dept",
            "url": "https://tkrcet.ac.in/mechanical-engineering/",
        },
        "civil": {
            "title": "Civil Dept",
            "url": "https://tkrcet.ac.in/civil-engineering/",
        },
        "aiml": {
            "title": "CSE-AIML Dept",
            "url": (
                "https://tkrcet.ac.in/"
                "cse-artificial-intelligence-machine-learning/"
            ),
        },
        "hostel": {
            "title": "Campus Life",
            "url": "https://tkrcet.ac.in/about-the-campus/",
        },
        "campus": {
            "title": "About Campus",
            "url": "https://tkrcet.ac.in/about-the-campus/",
        },
        "library": {
            "title": "Library",
            "url": "https://tkrcet.ac.in/library/",
        },
        "exam": {
            "title": "Academics",
            "url": "https://tkrcet.ac.in/academic-regulations/",
        },
        "calendar": {
            "title": "Academic Calendar",
            "url": "https://tkrcet.ac.in/academic-calendars/",
        },
        "ncc": {
            "title": "Campus Life",
            "url": "https://tkrcet.ac.in/about-the-campus/",
        },
        "nss": {
            "title": "Campus Life",
            "url": "https://tkrcet.ac.in/about-the-campus/",
        },
        "event": {
            "title": "TKRCET Home",
            "url": "https://tkrcet.ac.in/",
        },
        "fest": {
            "title": "TKRCET Home",
            "url": "https://tkrcet.ac.in/",
        },
        "naac": {
            "title": "NAAC",
            "url": "https://tkrcet.ac.in/naac-2/",
        },
        "alumni": {
            "title": "Alumni",
            "url": "https://tkrcet.ac.in/alumni-sub-domain/",
        },
        "transport": {
            "title": "About Campus",
            "url": "https://tkrcet.ac.in/about-the-campus/",
        },
        "mba": {
            "title": "MBA Department",
            "url": "https://tkrcet.ac.in/mba/",
        },
    }

    def get_topic_links(self, query):
        """Get relevant website links based on query keywords."""
        query_lower = query.lower()
        links = []
        seen_urls = set()

        for keyword, link_info in self.TOPIC_LINKS.items():
            if (
                keyword in query_lower
                and link_info["url"] not in seen_urls
            ):
                seen_urls.add(link_info["url"])
                links.append(link_info)

        if not links:
            links.append(
                {
                    "title": "TKRCET Website",
                    "url": "https://tkrcet.ac.in/",
                }
            )

        return links[:3]

    def extract_relevant_links(self, docs, query=""):
        """
        Extract URLs from documents,
        falling back to topic-based links.
        """

        links = []
        seen_urls = set()

        for doc in docs[:5]:
            url = doc.get("metadata", {}).get("url", "")
            source = doc.get("metadata", {}).get("source", "")

            link_url = (
                source
                if source and source.startswith("http")
                else url
            )

            if (
                link_url
                and link_url not in seen_urls
                and link_url.startswith("http")
            ):
                seen_urls.add(link_url)
                links.append(link_url)

        if len(links) < 3 and query:
            topic_links = self.get_topic_links(query)

            for tl in topic_links:
                if tl["url"] not in seen_urls:
                    seen_urls.add(tl["url"])
                    links.append(tl)

                    if len(links) >= 3:
                        break

        return links[:3]
