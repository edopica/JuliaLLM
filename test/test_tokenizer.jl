using JuliaLLM
using Test

@testset "Tokenizer" begin
    # Create a small dummy tokenizer for testing logic
    vocab = Dict(0=>"h", 1=>"e", 2=>"l", 3=>"o", 4=>"he", 5=>"ll", 6=>"hello", 7=>"Ġ", 8=>"Ġw")
    token_to_id = Dict(v=>k for (k,v) in vocab)
    merges = Dict(
        ("h", "e") => 0,
        ("l", "l") => 1,
        ("he", "ll") => 2,
        ("hell", "o") => 3,
        ("Ġ", "w") => 4
    )
    
    # Simple byte encoder mock (just for 'h', 'e', 'l', 'o', ' ', 'w')
    # In reality we use the full GPT-2 mapping.
    be = JuliaLLM.get_byte_encoder()
    bd = Dict(v=>k for (k,v) in be)
    
    tk = JuliaLLM.Tokenizer(vocab, token_to_id, merges, be, bd, 0, 0)
    
    @testset "Byte Encoding" begin
        # ' ' is 32, which maps to 'Ġ' (288)
        @test be[UInt8(' ')] == 'Ġ'
        @test bd['Ġ'] == UInt8(' ')
    end
    
    @testset "BPE Logic" begin
        # Test basic merging
        # "hello" -> ['h', 'e', 'l', 'l', 'o']
        # merge (h, e) -> ['he', 'l', 'l', 'o']
        # merge (l, l) -> ['he', 'll', 'o']
        # ... and so on depending on merges dict.
        # With our mock merges:
        # 1. (h,e) rank 0 -> "he", "l", "l", "o"
        # 2. (l,l) rank 1 -> "he", "ll", "o"
        # 3. (he, ll) rank 2 -> "hell", "o" (Wait, "hell" isn't in vocab, but BPE continues)
        # Actually, "hell" isn't in our vocab, so we need to be careful.
        
        # Let's use a simpler case
        simple_merges = Dict(
            ("h", "e") => 0,
            ("l", "l") => 1
        )
        tk_simple = JuliaLLM.Tokenizer(vocab, token_to_id, simple_merges, be, bd, 0, 0)
        ids = encode(tk_simple, "hello")
        # "hello" -> h, e, l, l, o
        # (h,e) -> he (id 4), l, l, o
        # (l,l) -> he, ll (id 5), o (id 3)
        @test ids == [4, 5, 3]
    end
    
    @testset "Decode" begin
        tk_simple = JuliaLLM.Tokenizer(vocab, token_to_id, merges, be, bd, 0, 0)
        @test decode(tk_simple, [4, 5, 3]) == "hello"
        @test decode(tk_simple, [7, 0]) == " h"
    end
end
