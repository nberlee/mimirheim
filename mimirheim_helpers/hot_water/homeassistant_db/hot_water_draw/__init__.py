"""Hot water draw forecast helper for mimirheim.

Estimates a household's hot water draw from the Home Assistant recorder
(tank temperatures and heat-source energy counters) and publishes it as a
timestamped forecast for a thermal boiler that uses ``use_draw_forecast``.
"""
