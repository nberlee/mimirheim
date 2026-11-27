"""Adapters that export one installation's data into the generic input set.

Each adapter reads one kind of source (a metrics query API, a home
automation recorder database, price and weather services). Which entities and URLs
they read comes from a local configuration file that is not committed.
"""
