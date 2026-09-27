# Independent data needed before stop-label promotion

No request has been sent externally.

Supply independently observed journeys on two different routes, both directions,
weekday/weekend and peak/offpeak, including sparse and truncated journeys:

- Actual dated visits: route, direction, vehicle, ordered visit position, stop ID,
  arrival/departure, observation source and timing uncertainty.
- Dated validator-to-vehicle assignments, including replacements and conflicts.
- Historical ordered patterns valid on the observed dates, including short turns
  and repeated stop visits; effective dates and provenance/known-at where available.
- For event-level accuracy: independently linked boarding/payment events. Nearest
  scheduled visit or manual interpretation of the same payment bursts is not gold.
- Source delivery/availability metadata and explicit missing-service/source windows.

Start with 20–50 independently checked journeys, then size the sample using
journey/day confidence intervals. Keep event-level, visit-level and count-level
truth separate. Hide evaluator labels from inference; disclose only predeclared
few-anchor inputs. Freeze hashes and train/development/external-evaluation splits
before tuning. The proposed G3 precision lower bound is a pilot goal, not achieved
quality or a guaranteed consequence of that sample size.
