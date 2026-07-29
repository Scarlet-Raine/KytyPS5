#ifndef KYTY_COMMON_LOGGING_LOG_H_
#define KYTY_COMMON_LOGGING_LOG_H_

#include "common/common.h"
#include "common/subsystems.h"

#include <atomic>
#include <cstdint>
#include <fmt/color.h>
#include <fmt/printf.h>
#include <string_view>

namespace Log {

KYTY_SUBSYSTEM_DEFINE(Log);

enum class Direction { Silent, Console, File };

Direction GetDirection();
bool      IsSilent();
void      Write(std::string_view text);
void      Write(fmt::text_style style, std::string_view text);
void      WriteFatal(std::string_view text);
void      WriteFatal(fmt::text_style style, std::string_view text);
void      Flush();

namespace Color {

inline constexpr auto Default       = fmt::text_style {};
inline constexpr auto Red           = fmt::fg(fmt::terminal_color::red);
inline constexpr auto Green         = fmt::fg(fmt::terminal_color::green);
inline constexpr auto Yellow        = fmt::fg(fmt::terminal_color::yellow);
inline constexpr auto Magenta       = fmt::fg(fmt::terminal_color::magenta);
inline constexpr auto Cyan          = fmt::fg(fmt::terminal_color::cyan);
inline constexpr auto White         = fmt::fg(fmt::terminal_color::white);
inline constexpr auto BrightRed     = fmt::fg(fmt::terminal_color::bright_red);
inline constexpr auto BrightGreen   = fmt::fg(fmt::terminal_color::bright_green);
inline constexpr auto BrightYellow  = fmt::fg(fmt::terminal_color::bright_yellow);
inline constexpr auto BrightMagenta = fmt::fg(fmt::terminal_color::bright_magenta);
inline constexpr auto BrightWhite   = fmt::fg(fmt::terminal_color::bright_white);

} // namespace Color

} // namespace Log

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOGF(...)                                                                                  \
	do {                                                                                           \
		if (!::Log::IsSilent()) {                                                                  \
			::Log::Write(::fmt::sprintf(__VA_ARGS__));                                             \
		}                                                                                          \
	} while (false)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOGF_COLOR(style, ...)                                                                     \
	do {                                                                                           \
		if (!::Log::IsSilent()) {                                                                  \
			::Log::Write((style), ::fmt::sprintf(__VA_ARGS__));                                    \
		}                                                                                          \
	} while (false)

// Log writes serialize every thread behind one sink, so an unbounded LOGF on a per-draw or
// per-job path can starve the GPU command processor outright (measured: presentation stalled
// until the guest's render-thread watchdog killed the process). Hot paths must bound their
// traffic: LOGF_BOUNDED keeps the first `limit` records from a call site.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOGF_BOUNDED(limit, ...)                                                                   \
	do {                                                                                           \
		static ::std::atomic<uint32_t> kyty_logf_count {0};                                        \
		if (kyty_logf_count.fetch_add(1, ::std::memory_order_relaxed) < (limit)) {                 \
			LOGF(__VA_ARGS__);                                                                     \
		}                                                                                          \
	} while (false)
// LOGF_SAMPLED additionally keeps every `every`-th record after the first `first`, so a
// runaway caller (e.g. a guest spinning on an API) stays visible without flooding the sink.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOGF_SAMPLED(first, every, ...)                                                           \
	do {                                                                                           \
		static ::std::atomic<uint64_t> kyty_logf_count {0};                                        \
		const auto kyty_logf_n = kyty_logf_count.fetch_add(1, ::std::memory_order_relaxed);        \
		if (kyty_logf_n < (first) || (kyty_logf_n % (every)) == 0) {                               \
			LOGF(__VA_ARGS__);                                                                     \
		}                                                                                          \
	} while (false)

#endif /* KYTY_COMMON_LOGGING_LOG_H_ */
