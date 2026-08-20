//! The protocol-independent half of the stream pipeline.
//!
//! Both wire protocols decode into their own event type and reassemble it with their own
//! accumulator, but everything between those two facts is identical: append an
//! end-of-transport sentinel, thread the accumulator through the stream, flatten each batch,
//! and yield an accumulator error in band as an `Err` item.
//!
//! Nothing here names either protocol. Each accumulator module implements
//! [`EventAccumulator`] for its own type, so a third protocol is a new module rather than an
//! edit to this one.
//!
//! That machinery is subtle enough to be worth having exactly once. The sentinel in
//! particular is a fix for a real defect — servers that stop sending without ever reporting
//! why would otherwise leave their content stranded in the accumulator's buffers and yield a
//! silently empty response — and a second transcription of it is a second place for that
//! defect to come back.

use std::pin::Pin;

use futures::stream::{Stream, StreamExt};

use crate::Result;
use crate::types::StreamEvent;

/// Reassembles one protocol's streaming events into [`StreamEvent`]s.
///
/// The two implementors are [`StreamAccumulator`](super::StreamAccumulator) for OpenAI chat
/// completions and [`AnthropicAccumulator`](super::AnthropicAccumulator) for Anthropic
/// messages. Both hold the same contract: [`Self::process`] never returns
/// [`StreamEvent::Finish`], and [`Self::finish`] always ends with exactly one.
pub trait EventAccumulator {
    /// The wire event this accumulator consumes.
    type Event;

    /// Consumes one event, returning anything it completed.
    fn process(&mut self, event: Self::Event) -> Result<Vec<StreamEvent>>;

    /// Drains remaining content and emits the terminating [`StreamEvent::Finish`].
    fn finish(&mut self) -> Result<Vec<StreamEvent>>;
}

/// Drives `events` through `accumulator`, yielding the [`StreamEvent`]s it produces.
///
/// The `None` appended after the event stream is the explicit end-of-transport signal, and
/// it is what emits the terminating [`StreamEvent::Finish`], so every stream reports how it
/// ended even when the server never said.
///
/// An accumulator error is yielded as an `Err` item and does not itself close the stream;
/// callers propagate it with `?` and stop pulling, which is what ends the response.
pub fn drive<A, S>(
    events: S,
    accumulator: A,
) -> Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>
where
    A: EventAccumulator + Send + 'static,
    A::Event: Send,
    S: Stream<Item = Result<A::Event>> + Send + 'static,
{
    let terminated = events.map(Some).chain(futures::stream::iter([None]));

    // `scan` yields one batch per input item; batches are then flattened into individual
    // events. An empty batch simply flattens to nothing, so events that only accumulate
    // state need no special-casing here.
    let flattened = terminated
        .scan(accumulator, |accumulator, item| {
            let batch = match item {
                Some(Ok(event)) => accumulator.process(event),
                Some(Err(error)) => Err(error),
                None => accumulator.finish(),
            };
            futures::future::ready(Some(batch))
        })
        .flat_map(|result| {
            futures::stream::iter(match result {
                Ok(events) => events.into_iter().map(Ok).collect(),
                Err(error) => vec![Err(error)],
            })
        });

    Box::pin(flattened)
}
