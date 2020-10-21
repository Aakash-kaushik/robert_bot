import os, re, codecs, csv, random
import unicodedata,itertools, torch
import torch.nn as nn
from torch import optim

corpus_name="cornell_movie_dialogs_corpus"
corpus=os.path.join("data",corpus_name)

def print_lines(file,n=10):
    """Shows some lines ftom the text file"""
    with open(file,'rb') as datafile:
        lines=datafile.readlines()
    for line in lines[:n]:
        print(line)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_lines(file,fields):
    """loads lines and splits then into fields and return a dict"""
    lines={}
    with open(file,"r",encoding="iso-8859-1") as f:
        for line in f:
            values=line.split(" +++$+++ ")
            line_obj={}
            for idx,field in enumerate(fields):
                line_obj[field]=values[idx]
            lines[line_obj[fields[0]]]=line_obj
    return lines

def load_conv(file,lines,fields):
    convs=[]
    with open(file,"r",encoding="iso-8859-1") as f:
        for line in f:
            values=line.split(" +++$+++ ")
            conv_obj={}
            for idx,field in enumerate(fields):
                conv_obj[field]=values[idx]
            line_id_pattern=re.compile('L[0-9]+')
            line_ids=line_id_pattern.findall(conv_obj[fields[-1]])
            conv_obj["lines"]=[]
            for line_id in line_ids:
                conv_obj["lines"].append(lines[line_id])
            convs.append(conv_obj)
    return convs

def sentence_pair_extract(convs):
    qa_pairs=[]
    for conv in convs:
        for i in range(len(conv["lines"])-1):
            input_line=conv["lines"][i]["text"].strip()
            target_line=conv["lines"][i+1]["text"].strip()

            if input_line and target_line:
                qa_pairs.append([input_line,target_line])
    return qa_pairs


#paths to files 
#movie_file=os.path.join(corpus,"movie_lines.txt")
#character_file=os.path.join(corpus,"movie_conversations.txt")
#formatted_file=os.path.join(corpus,"formatted_movie_lines.txt")

#delimiter="\t"
#delimiter=str(codecs.decode(delimiter,"unicode_escape"))

#lines=load_lines(movie_file,["lineID", "characterID", "movieID", "character", "text"])
#convs=load_conv(character_file,lines,["character1ID", "character2ID", "movieID", "utteranceIDs"])

#with open(formatted_file, "w", encoding="utf-8") as outputfile:
#    writer=csv.writer(outputfile,delimiter=delimiter,lineterminator="\n")
#    for pair in sentence_pair_extract(convs):
#        writer.writerow(pair)


#text_manip part
formatted_file=os.path.join(corpus,"formatted_movie_lines.txt")


pad_token=0
sos_token=1
eos_token=2

class Voc:
    def __init__(self,name):
        self.name=name
        self.trimmed=False
        self.word2idx={}
        self.word2count={}
        self.idx2word={pad_token:"PAD",sos_token:"SOS",eos_token:"EOS"}
        self.num_words=3

    def add_sentence(self,sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self,word):
        if word not in self.word2idx:
            self.word2idx[word]=self.num_words
            self.word2count[word]=1
            self.idx2word[self.num_words]=word
            self.num_words+=1
        else:
            self.word2count[word]+=1

    def trim(self,min_thr):
        if self.trimmed:
            return
        self.trimmed=True

        keep_word=[]

        for k,v in self.word2count.items():
            if v>=min_thr:
                keep_word.append(k)

        self.word2idx={}
        self.word2count={}
        self.idx2word={pad_token:"PAD",sos_token:"SOS",eos_token:"EOS"}
        self.num_words=3

        for word in keep_word:
            self.add_word(word)


max_length=10

def unicode_to_ascii(s):
    return "".join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!="Mn")


def normalize_string(s):
    s=unicode_to_ascii(s.lower().strip())
    s=re.sub(r"([.!?])",r" \1",s)
    s=re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    s=re.sub(r"\s+",r" ",s).strip()
    return s

def read_vocs(formatted_file,corpus_name):
    lines=open(formatted_file,encoding="utf-8").read().strip().split("\n")
    pairs=[[normalize_string(s) for s in l.split("\t")] for l in lines]
    voc=Voc(corpus_name)
    return voc, pairs

def filter_pair(p):
    """Returns True if pairs are smaller than max_length"""
    return len(p[0].split(" ")) <= max_length and len(p[1].split(" ")) <= max_length

def filter_pairs(pairs):
    """returns pairs of with filter_pair()==True"""
    return [pair for pair in pairs if filter_pair(pair)]

def load_prepare_data(corpus,corpus_name,formatted_file,save_dir):
    voc,pairs=read_vocs(formatted_file,corpus_name)
    pairs=filter_pairs(pairs)
    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])
    return voc, pairs

save_dir=os.path.join("data","save")
voc,pairs=load_prepare_data(corpus,corpus_name,formatted_file,save_dir)


def trim_rare_words(voc,pairs,min_count):
    voc.trim(min_count)

    keep_pairs=[]
    for pair in pairs:
        input_sentence=pair[0]
        output_sentence=pair[1]
        keep_input=True
        keep_output=True
        for word in input_sentence.split(" "):
            if word not in voc.word2idx:
                keep_input=False
                break
        for word in output_sentence.split(" "):
            if word not in voc.word2idx:
                keep_output=False
                break

        if keep_output and keep_input:
            keep_pairs.append(pair)

    return keep_pairs

min_count=3     #pairs with word_count less than min_count will be removedby trim_rare_words() 
pairs=trim_rare_words(voc,pairs,min_count)

def indexes_from_sentence(voc,sentence):
    return [voc.word2idx[word] for word in sentence.split(" ")]+[eos_token]

def zero_padding(l,fillval=pad_token):
    return list(itertools.zip_longest(*l,fillvalue=fillval))

def binary_matrix(l,val=pad_token):
    m=[]
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token==pad_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def input_var(l,voc):
    indexes_batch=[indexes_from_sentence(voc,sentence) for sentence in l]
    lengths=torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list=zero_padding(indexes_batch)
    pad_var=torch.LongTensor(pad_list)
    return pad_var,lengths

def output_var(l,voc):
    indexes_batch=[indexes_from_sentence(voc,sentence) for sentence in l]
    max_target_length=max([len(indexes) for indexes in indexes_batch])
    pad_list=zero_padding(indexes_batch)
    mask=binary_matrix(pad_list)
    mask=torch.BoolTensor(mask)
    pad_var=torch.LongTensor(pad_list)
    return pad_var,mask,max_target_length

def batch2train_data(voc,pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")),reverse=True)
    input_batch,output_batch=[],[]
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp,lengths=input_var(input_batch,voc)
    output,mask,max_target_length=output_var(output_batch,voc)
    return inp,lengths,output,mask,max_target_length


#model part
class encoder_rnn(nn.Module):
    def __init__(self,hidden_size,embedding,n_layers=1,dropout=0):
        super(encoder_rnn,self).__init__()
        self.n_layers=n_layers
        self.hidden_size=hidden_size
        self.embedding=embedding

        self.gru=nn.GRU(hidden_size,hidden_size, n_layers,dropout=(0 if n_layers==1 else dropout),bidirectional=True)

    def forward(self,input_seq,input_lengths,hidden=None):
        embedded=self.embedding(input_seq)
        packed=nn.utils.rnn.pack_padded_sequence(embedded,input_lengths)
        outputs,hidden=self.gru(packed,hidden)
        outputs,_=nn.utils.rnn.pad_packed_sequence(outputs)
        outputs=outputs[:,:,:self.hidden_size]+outputs[:,:,self.hidden_size:]
        return outputs,hidden

class Attn(nn.Module):
    def __init__(self,method,hidden_size):
        super(Attn,self).__init__()
        self.method=method 
        if self.method not in ["dot","general","concat"]:
            raise ValueError(self.method,"is not an appropriate attention method")
        self.hidden_size=hidden_size
        if self.method=="general":
            self.attn=nn.Linear(self.hidden_size,hidden_size)
        elif self.method=="concat":
            self.attn=nn.Linear(self.hidden_size*2,hidden_size)
            self.v=nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self,hidden,encoder_output):
        return torch.sum(hidden,encoder_output,dim=2)

    def general_score(self,hidden,encoder_output):
        energy=self.attn(encoder_output)
        return torch.sum(energy*hidden,dim=2)

    def concat_score(self,hidden,encoder_output):
        energy=self.attn(torch.cat((hidden.expand(encoder_output.size(0),-1,-1),encoder_output),2)).tanh()
        return torch.sum(self.v*energy,dim=2)

    def forward(self,hidden,encoder_outputs):
        if self.method=="general":
            attn_energies=self.general_score(hidden,encoder_outputs)
        elif self.method=="concat":
            attn_energies=self.concat_score(hidden,encoder_outputs)
        elif self.method=="dot":
            attn_energies=self.dot_score(hidden,encoder_outputs)

        attn_energies=attn_energies.t()

        return nn.functional.softmax(attn_energies,dim=1).unsqueeze(1)

class attn_decoder_rnn(nn.Module):
    def __init__(self,attn_model,embedding,hidden_size,output_size,n_layers=1,dropout=0.1):
        super(attn_decoder_rnn,self).__init__()
        self.attn_model=attn_model
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.n_layers=n_layers
        self.dropout=dropout

        self.embedding=embedding
        self.embedding_dropout=nn.Dropout(dropout)
        self.gru=nn.GRU(hidden_size,hidden_size,n_layers,dropout=(0 if n_layers==1 else dropout))
        self.concat=nn.Linear(hidden_size*2,hidden_size)
        self.out=nn.Linear(hidden_size,output_size)
        self.attn=Attn(attn_model,hidden_size)

    def forward(self,input_step,last_hidden,encoder_outputs):
        embedded=self.embedding(input_step)
        embedded=self.embedding_dropout(embedded)
        rnn_output,hidden=self.gru(embedded,last_hidden)
        attn_weights=self.attn(rnn_output,encoder_outputs)
        context=attn_weights.bmm(encoder_outputs.transpose(0,1))
        rnn_output=rnn_output.squeeze(0)
        context=context.squeeze(1)
        concat_input=torch.cat((rnn_output,context),1)
        concat_output=torch.tanh(self.concat(concat_input))
        output=self.out(concat_output)
        output=nn.functional.softmax(output,dim=1)
        
        return output,hidden

#train part
def mask_nll_loss(inp,target,mask):
    n_total=mask.sum()
    cross_entropy = -torch.log(torch.gather(inp,1,target.view(-1,1)).squeeze(1))
    loss=cross_entropy.masked_select(mask).mean()
    loss=loss.to(device)
    return loss, n_total.item()

def train(input_variable,lengths,target_variable, mask,max_target_length,encoder,decoder,embedding,
         encoder_optimizer,decoder_optimizer,batch_size,clip,max_length=15):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable=input_variable.to(device)
    lengths=lengths.to(device)
    target_variable=target_variable.to(device)
    mask=mask.to(device)

    loss=0
    print_losses=[]
    n_totals=0

    encoder_outputs, encoder_hidden = encoder(input_variable,lengths)

    decoder_input=torch.LongTensor([[sos_token for _ in range(batch_size)]])
    decoder_input=decoder_input.to(device)

    decoder_hidden=encoder_hidden[:decoder.n_layers]

    use_teacher_forcing= True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(max_target_length):
            decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden,encoder_outputs)
            decoder_input=target_variable[t].view(1,-1)
            mask_loss,n_total=mask_nll_loss(decoder_output,target_variable[t],mask[t])
            loss+=mask_loss
            print_losses.append(mask_loss.item()*n_total)
            n_totals += n_total

    else:
        for t in range(max_target_length):
            decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden,encoder_outputs)
            _,topi=decoder_output.topk(1)
            decoder_input=torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input=decoder_input.to(device)
            mask_loss,n_total=mask_nll_loss(decoder_output,target_variable[t],mask[t])
            loss+=mask_loss
            print_losses.append(mask_loss.item()*n_total)
            n_totals+=n_total

    loss.backward()

    _=torch.nn.utils.clip_grad_norm_(encoder.parameters(),clip)
    _=torch.nn.utils.clip_grad_norm_(decoder.parameters(),clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses)/n_totals

def train_iters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers,
                decoder_n_layers, save_dir, n_iter, batch_size, print_every, save_every, clip, corpus_name, load_filename):
    training_batches=[batch2train_data(voc,[random.choice(pairs) for _ in range(batch_size)]) for _ in range(n_iteration)]
    print("initializing")
    start_iter=1
    print_loss=0
    if load_filename:
        start_iter=checkpoint['iteration']+1

    print('Training')
    for iteration in range(start_iter,n_iteration+1):
        training_batch=training_batches[iteration-1]
        input_variable,lengths,target_variable,mask,max_target_length=training_batch
        loss=train(input_variable,lengths,target_variable,mask,max_target_length,encoder,decoder,embedding,encoder_optimizer,decoder_optimizer,batch_size,clip)
        print_loss+=loss

        if iteration % print_every == 0:
            print_loss_avg=print_loss/print_every
            print("iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration,iteration/n_iteration*100,print_loss_avg))
            print_loss=0

        if iteration % save_every == 0:
            directory=os.path.join(save_dir,model_name,corpus_name,'{}-{}_{}'.format(encoder_n_layers,decoder_n_layers,hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({"iteration":iteration,
                        "en":encoder.state_dict(),
                        "de":decoder.state_dict(),
                        "en_opt":encoder_optimizer.state_dict(),
                        "de_opt":decoder_optimizer.state_dict(),
                        "loss":loss,
                        "voc_dict":voc.__dict__,
                        "embedding":embedding.state_dict()
                       },os.path.join(directory,"{}_{}.tar".format(iteration,"checkpoint")))


#eval part
class greedy_search_decoder(nn.Module):
    def __init__(self,encoder,decoder):
        super(greedy_search_decoder,self).__init__()
        self.encoder=encoder
        self.decoder=decoder

    def forward(self,input_seq,input_length,max_length):
        encoder_outputs,encoder_hidden=self.encoder(input_seq,input_length)
        decoder_hidden=encoder_hidden[:decoder.n_layers]
        decoder_input=torch.ones(1,1,device=device,dtype=torch.long)* sos_token
        all_tokens=torch.zeros([0],device=device,dtype=torch.long)
        all_scores=torch.zeros([0],device=device)     
        for _ in range(max_length):
            decoder_output,decoder_hidden=self.decoder(decoder_input,decoder_hidden,encoder_outputs)
            decoder_scores,decoder_input=torch.max(decoder_output,dim=1)
            all_tokens=torch.cat((all_tokens,decoder_input),dim=0)
            all_scores=torch.cat((all_scores,decoder_scores),dim=0)
            decoder_input=torch.unsqueeze(decoder_input,0)
        return all_tokens,all_scores

def evaluate(encoder,decoder,searcher,voc,sentence,max_length=max_length):
    indexes_batch=[indexes_from_sentence(voc,sentence)]
    lengths=torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch=torch.LongTensor(indexes_batch).transpose(0,1)
    input_batch=input_batch.to(device)
    lengths=lengths.to(device)
    tokens,scores=searcher(input_batch,lengths,max_length)
    decoded_words=[voc.idx2word[token.item()] for token in tokens]
    return decoded_words

def eval_input(encoder,decoder,searcher,voc,input_sentence="hey"):
    try:
        input_sentence=normalize_string(input_sentence)
        output_words=evaluate(encoder,decoder,searcher,voc,input_sentence)
        output_words[:]=[x for x in output_words if not(x=='EOS' or x=='PAD')]
    except KeyError:
        output_words = -1
    return output_words


#eval
model_name = 'cb_model'
attn_model = 'dot'
attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = r"5000_checkpoint.tar"
checkpoint_iter = 5000

if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = encoder_rnn(hidden_size, embedding, encoder_n_layers, dropout)
decoder = attn_decoder_rnn(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
#print('Robert: Hi, I am Robert')

encoder.eval()
decoder.eval()

# Initialize search module
searcher = greedy_search_decoder(encoder, decoder)
