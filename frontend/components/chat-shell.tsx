"use client";

import {
  ChangeEvent,
  FormEvent,
  KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
  useTransition,
} from "react";
import Image from "next/image";
import Link from "next/link";
import logo from "../images/logo.png";
import { askQuestion, ChatResponse, fetchStatus, SystemStatus } from "../lib/api";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: ChatResponse;
};

const MATTER_TYPES = [
  "Family law",
  "Employment",
  "Contracts",
  "Property",
  "Business",
  "Immigration",
  "Civil dispute",
  "Other",
];

const URGENCY_LEVELS = ["General guidance", "This week", "Urgent deadline", "Court or agency notice"];

const CONSULT_STARTERS = [
  "What should I prepare before meeting a lawyer?",
  "Can you explain my options in plain language?",
  "What questions should I ask during a consultation?",
];

const SendIcon = () => (
  <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <path d="M2.01 21 23 12 2.01 3 2 10l15 2-15 2z" />
  </svg>
);

const ScaleIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
    <path d="M12 3v18M5 6h14M7 6l-4 8h8L7 6zM17 6l-4 8h8l-4-8z" />
  </svg>
);

const ShieldIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
    <path d="M12 2 5 6v5c0 5 3 9 7 11 4-2 7-6 7-11V6l-7-4z" />
    <path d="m9 12 2 2 4-4" />
  </svg>
);

const ClockIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3 2" />
  </svg>
);

const ErrorIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
    <circle cx="12" cy="12" r="10" />
    <path d="M12 8v4M12 16h.01" />
  </svg>
);

const BrandLogo = () => <Image alt="Synkora AI" className="brand-logo-image" priority src={logo} />;

const SynkoraLogo = () => (
  <svg viewBox="0 0 18 18" fill="white" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <path
      d="M9 1.5C5.5 1.5 2.5 4.5 2.5 8c0 2 .9 3.8 2.3 5l-.8 2.5 2.5-.8C7.2 15.1 8.1 15.5 9 15.5c3.5 0 6.5-3 6.5-6.5S12.5 1.5 9 1.5z"
      opacity=".2"
    />
    <path d="M9 2.5C6 2.5 3.5 5 3.5 8S6 13.5 9 13.5c.8 0 1.5-.2 2.2-.5l2.3.8-.8-2.3c.5-.8.8-1.7.8-2.5C13.5 5 11 2.5 9 2.5zm0 1.5c2.2 0 4 1.8 4 4S11.2 12 9 12c-.7 0-1.4-.2-2-.6l-.2-.1-1.4.5.5-1.4-.1-.2C5.2 9.6 5 8.8 5 8.1c0-2.3 1.8-4.1 4-4.1z" />
  </svg>
);

function buildConsultationPrompt(values: {
  matterType: string;
  urgency: string;
  jurisdiction: string;
  summary: string;
}): string {
  return [
    "Legal consultation intake:",
    `Matter type: ${values.matterType || "Not specified"}`,
    `Urgency: ${values.urgency || "Not specified"}`,
    `Jurisdiction/location: ${values.jurisdiction || "Not specified"}`,
    `Client concern: ${values.summary}`,
    "",
    "Give a clear first consultation response. Explain likely legal issues, practical next steps, documents or facts to prepare, and questions the client should ask an attorney. Do not claim to be a lawyer or provide a final legal opinion.",
  ].join("\n");
}

export function ChatShell() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [matterType, setMatterType] = useState(MATTER_TYPES[0]);
  const [urgency, setUrgency] = useState(URGENCY_LEVELS[0]);
  const [jurisdiction, setJurisdiction] = useState("");
  const [summary, setSummary] = useState("");
  const [followUp, setFollowUp] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const chatEndRef = useRef<HTMLDivElement>(null);
  const summaryRef = useRef<HTMLTextAreaElement>(null);
  const followUpRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const sys = await fetchStatus();
        if (active) setStatus(sys);
      } catch (err) {
        if (active) setError(err instanceof Error ? err.message : "Failed to load consultation status.");
      }
    };
    void refresh();
    const interval = setInterval(() => void refresh(), 10000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isPending]);

  const consultationReady = (status?.ready_document_count ?? 0) > 0;
  const latestAssistant = useMemo(() => [...messages].reverse().find((message) => message.role === "assistant"), [messages]);

  const submitConsultation = (rawQuestion: string, displayText: string) => {
    const cleaned = rawQuestion.trim();
    if (!cleaned) return;
    if (!consultationReady) {
      setError("The consultation knowledge base is not ready yet. Please try again shortly.");
      return;
    }

    setError(null);
    const userMsg: Message = {
      id: `u-${Date.now()}`,
      role: "user",
      content: displayText,
    };
    setMessages((current) => [...current, userMsg]);

    startTransition(async () => {
      try {
        const response = await askQuestion(cleaned, conversationId);
        setConversationId(response.conversation_id);
        setMessages((current) => [
          ...current,
          {
            id: `a-${Date.now()}`,
            role: "assistant",
            content: response.answer,
            response,
          },
        ]);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Something went wrong.");
      }
    });
  };

  const handleConsultSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const cleanedSummary = summary.trim();
    if (cleanedSummary.length < 12) {
      setError("Please describe your legal concern in a little more detail.");
      return;
    }

    submitConsultation(
      buildConsultationPrompt({
        matterType,
        urgency,
        jurisdiction: jurisdiction.trim(),
        summary: cleanedSummary,
      }),
      cleanedSummary,
    );
  };

  const handleFollowUpSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const cleaned = followUp.trim();
    if (!cleaned) return;
    setFollowUp("");
    if (followUpRef.current) followUpRef.current.style.height = "auto";
    submitConsultation(cleaned, cleaned);
  };

  const handleStarter = (starter: string) => {
    setSummary(starter);
    requestAnimationFrame(() => {
      summaryRef.current?.focus();
      if (summaryRef.current) {
        summaryRef.current.style.height = "auto";
        summaryRef.current.style.height = `${Math.min(summaryRef.current.scrollHeight, 220)}px`;
      }
    });
  };

  const handleTextareaResize = (event: ChangeEvent<HTMLTextAreaElement>) => {
    const textarea = event.target;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 220)}px`;
  };

  const handleFollowUpKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  return (
    <div className="consult-root">
      <nav className="consult-nav">
        <Link className="consult-logo-link" href="/" aria-label="Synkora AI legal consultation">
          <div className="logo-mark logo-mark--brand">
            <BrandLogo />
          </div>
        </Link>

        <div className="consult-nav-meta">
          <span className={`consult-status ${consultationReady ? "is-ready" : ""}`}>
            {consultationReady ? "Consultation ready" : "Preparing consultation"}
          </span>
          <Link className="admin-nav-link" href="/admin">
            Admin
          </Link>
        </div>
      </nav>

      <main className="consult-main">
        <section className="consult-intake" aria-labelledby="consult-title">
          <div className="consult-hero">
            <p className="consult-eyebrow">Confidential legal intake</p>
            <h1 id="consult-title">Start with a clear legal consultation.</h1>
            <p>
              Share the situation, deadlines, and location. Synkora organizes your concern into practical first steps for a lawyer conversation.
            </p>

            <div className="consult-trust-row" aria-label="Consultation safeguards">
              <span>
                <ShieldIcon />
                Private intake
              </span>
              <span>
                <ClockIcon />
                Deadline aware
              </span>
              <span>
                <ScaleIcon />
                Attorney-ready summary
              </span>
            </div>
          </div>

          <form className="consult-form" onSubmit={handleConsultSubmit}>
            <div className="consult-field-grid">
              <label className="consult-field">
                <span>Matter type</span>
                <select value={matterType} onChange={(event) => setMatterType(event.target.value)}>
                  {MATTER_TYPES.map((item) => (
                    <option key={item}>{item}</option>
                  ))}
                </select>
              </label>

              <label className="consult-field">
                <span>Urgency</span>
                <select value={urgency} onChange={(event) => setUrgency(event.target.value)}>
                  {URGENCY_LEVELS.map((item) => (
                    <option key={item}>{item}</option>
                  ))}
                </select>
              </label>
            </div>

            <label className="consult-field">
              <span>Jurisdiction or location</span>
              <input
                value={jurisdiction}
                onChange={(event) => setJurisdiction(event.target.value)}
                placeholder="Example: Colombo, Sri Lanka or California, USA"
              />
            </label>

            <label className="consult-field">
              <span>What happened?</span>
              <textarea
                ref={summaryRef}
                value={summary}
                onChange={(event) => {
                  setSummary(event.target.value);
                  handleTextareaResize(event);
                }}
                placeholder="Describe the issue, dates, parties involved, notices received, and what outcome you want."
                rows={5}
              />
            </label>

            <div className="consult-form-footer">
              <button className="consult-primary-btn" disabled={isPending || !summary.trim()} type="submit">
                {isPending && messages.length === 0 ? "Preparing..." : "Start consultation"}
              </button>
              <p>This is general information for consultation prep, not a final legal opinion.</p>
            </div>
          </form>

          <div className="consult-starters" aria-label="Example consultation prompts">
            {CONSULT_STARTERS.map((starter) => (
              <button key={starter} onClick={() => handleStarter(starter)} type="button">
                {starter}
              </button>
            ))}
          </div>
        </section>

        <section className="consult-conversation" aria-label="Consultation response">
          <div className="consult-response-head">
            <div>
              <p className="consult-eyebrow">Consultation workspace</p>
              <h2>{latestAssistant ? "Response" : "Your guidance will appear here"}</h2>
            </div>
            {latestAssistant?.response && (
              <span className="consult-confidence">
                {Math.round(latestAssistant.response.confidence * 100)}% confidence
              </span>
            )}
          </div>

          <div className="consult-response-body">
            {messages.length === 0 ? (
              <div className="consult-empty-state">
                <ScaleIcon />
                <h3>Tell us the facts first.</h3>
                <p>
                  A focused intake helps separate deadlines, evidence, parties, and likely next steps before a lawyer reviews the matter.
                </p>
              </div>
            ) : (
              <div className="messages consult-messages">
                {messages.map((message) =>
                  message.role === "user" ? (
                    <div className="msg-user" key={message.id}>
                      <div className="msg-user-bubble">{message.content}</div>
                    </div>
                  ) : (
                    <div className="msg-assistant" key={message.id}>
                      <div className="msg-assistant-avatar">
                        <SynkoraLogo />
                      </div>
                      <div className="msg-assistant-body">
                        {message.response && (
                          <div className="msg-chips">
                            <span className={`msg-chip ${message.response.grounded ? "chip-grounded" : "chip-ungrounded"}`}>
                              {message.response.grounded ? "Knowledge-base supported" : "Needs attorney review"}
                            </span>
                          </div>
                        )}
                        <p className="msg-answer">{message.content}</p>
                      </div>
                    </div>
                  ),
                )}

                {isPending && (
                  <div className="msg-pending msg-assistant">
                    <div className="msg-assistant-avatar">
                      <SynkoraLogo />
                    </div>
                    <div className="typing-indicator" aria-label="Preparing response">
                      <div className="typing-dot" />
                      <div className="typing-dot" />
                      <div className="typing-dot" />
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
            )}
          </div>

          {messages.length > 0 && (
            <form className="consult-followup" onSubmit={handleFollowUpSubmit}>
              {error && (
                <div className="error-banner">
                  <ErrorIcon />
                  {error}
                </div>
              )}
              <div className="consult-followup-box">
                <textarea
                  ref={followUpRef}
                  value={followUp}
                  onChange={(event) => {
                    setFollowUp(event.target.value);
                    handleTextareaResize(event);
                  }}
                  onKeyDown={handleFollowUpKeyDown}
                  placeholder="Ask a follow-up about deadlines, evidence, risks, or next steps..."
                  rows={1}
                  aria-label="Ask a follow-up question"
                />
                <button className="send-btn" disabled={isPending || !followUp.trim()} type="submit" title="Send">
                  <SendIcon />
                </button>
              </div>
              <p className="composer-hint">Press Enter to send. Shift+Enter for a new line.</p>
            </form>
          )}

          {messages.length === 0 && error && (
            <div className="error-banner consult-inline-error">
              <ErrorIcon />
              {error}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
